%% ═══════════════════════════════════════════════════════════════════════════
%  1_3_detect_circles.m
%
%  Circle Detection on Drift-Corrected Frames (MATLAB version)
%  Replaces Python 1_3_detect_circles.py — uses MATLAB's imfindcircles
%
%  Pipeline per video folder:
%    1. Load frame_001.png → adapthisteq → medfilt2 → threshold (binary mask)
%    2. Pass 1: wide Hough [30,50] to estimate mean radius
%    3. Pass 2: tight Hough [meanR-5, meanR+5] on binary mask
%    4. Non-max suppression to remove duplicates
%    5. Geometry cleanup: remove edge-cut circles, shrink overlapping ones
%    6. Consistent numbering (left→right, top→bottom)
%    7. Annotate ALL 30 frames with colored circles:
%       green = ok, yellow = shrunk, red = edge-removed
%    8. Save logs (CSV, JSON, plots) — matches Python output structure
%
%  Input:  D:\Research\Cancer_Cell_Analysis\extracted_frames\image_frames\
%  Output: D:\Research\Cancer_Cell_Analysis\image_circle\images\<video>\
%          D:\Research\Cancer_Cell_Analysis\image_circle\logs\<video>\
%
%  Author: Based on Antardip Himel's MATLAB pipeline
%  Date:   February 2026
% ═══════════════════════════════════════════════════════════════════════════
clc; clear; close all;

%% ─── CONFIGURATION ──────────────────────────────────────────────────────
INPUT_DIR  = 'D:\Research\Cancer_Cell_Analysis\extracted_frames\image_frames';
OUTPUT_DIR = 'D:\Research\Cancer_Cell_Analysis\image_circle';

% Preprocessing
CLAHE_CLIP   = 0.02;       % adapthisteq ClipLimit
CLAHE_TILES  = [8 8];      % adapthisteq NumTiles
MEDIAN_KSIZE = [5 5];      % medfilt2 kernel
THRESH_VALUE = 200;         % Binary threshold (S0 > 200)

% Hough — Pass 1 (wide radius estimation)
PASS1_RAD_RANGE = [30 50];

% Hough — Pass 2 (tight around mean)
PASS2_RAD_TOL = 5;

% Hough tuning
HOUGH_SENSITIVITY   = 1.0;
HOUGH_EDGE_THRESH   = 0.01;  % tight pass uses lower edge threshold
HOUGH_METHOD        = 'TwoStage';
HOUGH_POLARITY      = 'dark';

% NMS
NMS_DIST_FACTOR = 1.6;

% Annotation colors (RGB 0-255 for insertShape)
COLOR_OK     = [0   200 0  ];   % Green
COLOR_SHRUNK = [220 220 0  ];   % Yellow
COLOR_EDGE   = [220 0   0  ];   % Red

%% ─── DISCOVER VIDEO FOLDERS ─────────────────────────────────────────────
if ~exist(INPUT_DIR, 'dir')
    error('Input directory not found: %s', INPUT_DIR);
end

allDirs = dir(INPUT_DIR);
allDirs = allDirs([allDirs.isdir]);
allDirs = allDirs(~ismember({allDirs.name}, {'.', '..'}));

if isempty(allDirs)
    error('No video folders found in: %s', INPUT_DIR);
end

folderNames = {allDirs.name};
numFolders  = numel(folderNames);

%% ─── FOLDER SELECTION UI ───────────────────────────────────────────────
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  CIRCLE DETECTION — Folder Selection\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');
fprintf('  Available video folders:\n\n');
fprintf('  [0]  ➜  ALL folders (%d total)\n\n', numFolders);
for k = 1:numFolders
    fprintf('  [%d]  %s\n', k, folderNames{k});
end
fprintf('\n');

selInput = input('  Enter folder numbers (e.g. 1 3 5) or 0 for ALL: ', 's');
selInput = strtrim(selInput);

if isempty(selInput) || strcmp(selInput, '0')
    selectedIdx = 1:numFolders;
    fprintf('  ➜ Processing ALL %d folders\n\n', numFolders);
else
    selectedIdx = str2num(selInput); %#ok<ST2NM>
    selectedIdx = unique(selectedIdx(selectedIdx >= 1 & selectedIdx <= numFolders));
    if isempty(selectedIdx)
        error('No valid folder numbers entered.');
    end
    fprintf('  ➜ Processing %d folder(s):', numel(selectedIdx));
    for k = selectedIdx
        fprintf(' [%d]%s', k, folderNames{k});
    end
    fprintf('\n\n');
end

%% ─── MAIN PROCESSING LOOP ──────────────────────────────────────────────
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  CIRCLE DETECTION PIPELINE (MATLAB)\n');
fprintf('  %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('  Input:  %s\n', INPUT_DIR);
fprintf('  Output: %s\n\n', OUTPUT_DIR);
fprintf('───────────────────────────────────────────────────────────────\n');

totalProcessed = 0;
totalSkipped   = 0;

for si = 1:numel(selectedIdx)
    idx = selectedIdx(si);
    videoName = folderNames{idx};
    inputPath = fullfile(INPUT_DIR, videoName);

    fprintf('\n▶ [%d/%d] %s\n', si, numel(selectedIdx), videoName);

    try
        processOneVideo(videoName, inputPath, OUTPUT_DIR, ...
            CLAHE_CLIP, CLAHE_TILES, MEDIAN_KSIZE, THRESH_VALUE, ...
            PASS1_RAD_RANGE, PASS2_RAD_TOL, ...
            HOUGH_SENSITIVITY, HOUGH_EDGE_THRESH, HOUGH_METHOD, HOUGH_POLARITY, ...
            NMS_DIST_FACTOR, COLOR_OK, COLOR_SHRUNK, COLOR_EDGE);
        totalProcessed = totalProcessed + 1;
    catch ME
        fprintf('   ❌ ERROR: %s\n\n', ME.message);
        totalSkipped = totalSkipped + 1;
    end
end

%% ─── FINAL SUMMARY ─────────────────────────────────────────────────────
fprintf('\n═══════════════════════════════════════════════════════════════\n');
fprintf('  🎉 Circle detection complete!\n');
fprintf('  Processed: %d | Skipped: %d\n', totalProcessed, totalSkipped);
fprintf('  Annotated frames → %s\n', fullfile(OUTPUT_DIR, 'images'));
fprintf('  Detection logs   → %s\n', fullfile(OUTPUT_DIR, 'logs'));
fprintf('  Circle colors: 🟢 OK  🟡 Shrunk  🔴 Edge-cut\n');
fprintf('═══════════════════════════════════════════════════════════════\n');


%% ═══════════════════════════════════════════════════════════════════════
%  FUNCTIONS
%  ═══════════════════════════════════════════════════════════════════════

function processOneVideo(videoName, inputPath, OUTPUT_DIR, ...
    CLAHE_CLIP, CLAHE_TILES, MEDIAN_KSIZE, THRESH_VALUE, ...
    PASS1_RAD_RANGE, PASS2_RAD_TOL, ...
    HOUGH_SENSITIVITY, HOUGH_EDGE_THRESH, HOUGH_METHOD, HOUGH_POLARITY, ...
    NMS_DIST_FACTOR, COLOR_OK, COLOR_SHRUNK, COLOR_EDGE)

    % Output directories (matching Python structure)
    outImages = fullfile(OUTPUT_DIR, 'images', videoName);
    outLogs   = fullfile(OUTPUT_DIR, 'logs',   videoName);
    if ~exist(outImages, 'dir'), mkdir(outImages); end
    if ~exist(outLogs,   'dir'), mkdir(outLogs);   end

    % Skip if already processed
    if exist(fullfile(outLogs, 'detection_metadata.json'), 'file')
        fprintf('   ⏭ Already processed — skipping. Delete logs to reprocess.\n\n');
        return;
    end

    % ─── Find frame files ────────────────────────────────────────────
    frameFiles = dir(fullfile(inputPath, 'frame_*.png'));
    if isempty(frameFiles)
        % Also check for .jpg
        frameFiles = [dir(fullfile(inputPath, 'frame_*.png')); ...
                      dir(fullfile(inputPath, 'frame_*.jpg'))];
    end
    if isempty(frameFiles)
        fprintf('   ⚠ No frames found — skipping.\n\n');
        return;
    end
    frameFiles = natsortfiles({frameFiles.name});  % natural sort
    numFrames  = numel(frameFiles);

    % ─── Load first frame ────────────────────────────────────────────
    firstPath = fullfile(inputPath, frameFiles{1});
    I0 = imread(firstPath);
    [H, W, ~] = size(I0);
    G0 = rgb2gray(I0);
    fprintf('   Loaded %d frames (%dx%d)\n', numFrames, W, H);

    % ═════════════════════════════════════════════════════════════════
    % STEP 1: Preprocess → binary mask
    % ═════════════════════════════════════════════════════════════════
    fprintf('   [1/6] Preprocessing → binary mask (threshold=%d)...\n', THRESH_VALUE);
    E0 = adapthisteq(G0, 'ClipLimit', CLAHE_CLIP, 'NumTiles', CLAHE_TILES);
    S0 = medfilt2(E0, MEDIAN_KSIZE);
    mask0 = uint8(S0 > THRESH_VALUE) * 255;

    % Save QC images
    imwrite(E0,    fullfile(outLogs, 'qc_01_enhanced.png'));
    imwrite(S0,    fullfile(outLogs, 'qc_02_smoothed.png'));
    imwrite(mask0, fullfile(outLogs, 'qc_03_binary_mask.png'));

    % ═════════════════════════════════════════════════════════════════
    % STEP 2: Pass 1 — estimate mean radius
    % ═════════════════════════════════════════════════════════════════
    fprintf('   [2/6] Pass 1: estimating mean radius [r=%d-%d]...\n', ...
        PASS1_RAD_RANGE(1), PASS1_RAD_RANGE(2));

    [~, r0] = imfindcircles(mask0, PASS1_RAD_RANGE, ...
        'ObjectPolarity', HOUGH_POLARITY, ...
        'Sensitivity',    HOUGH_SENSITIVITY, ...
        'EdgeThreshold',  0.1, ...
        'Method',         HOUGH_METHOD);

    if isempty(r0)
        fprintf('   ⚠ No circles in pass 1 — skipping.\n\n');
        return;
    end
    meanRadius = round(mean(r0));
    fprintf('   Pass 1: %d circles, mean radius = %dpx\n', numel(r0), meanRadius);

    % ═════════════════════════════════════════════════════════════════
    % STEP 3: Pass 2 — tight detection
    % ═════════════════════════════════════════════════════════════════
    rMin = max(10, meanRadius - PASS2_RAD_TOL);
    rMax = meanRadius + PASS2_RAD_TOL;
    fprintf('   [3/6] Pass 2: tight detection [r=%d-%d]...\n', rMin, rMax);

    [centers, radii] = imfindcircles(mask0, [rMin rMax], ...
        'ObjectPolarity', HOUGH_POLARITY, ...
        'Sensitivity',    HOUGH_SENSITIVITY, ...
        'EdgeThreshold',  HOUGH_EDGE_THRESH, ...
        'Method',         HOUGH_METHOD);

    if isempty(centers)
        % Fallback: wider range
        fprintf('   ⚠ No circles in pass 2 — trying wider range...\n');
        rMin2 = max(10, meanRadius - 10);
        rMax2 = meanRadius + 10;
        [centers, radii] = imfindcircles(mask0, [rMin2 rMax2], ...
            'ObjectPolarity', HOUGH_POLARITY, ...
            'Sensitivity',    HOUGH_SENSITIVITY, ...
            'EdgeThreshold',  HOUGH_EDGE_THRESH, ...
            'Method',         HOUGH_METHOD);
        if isempty(centers)
            fprintf('   ⚠ Still no circles — skipping.\n\n');
            return;
        end
    end
    fprintf('   Pass 2: %d circles detected\n', size(centers,1));

    % ═════════════════════════════════════════════════════════════════
    % STEP 4: Non-max suppression
    % ═════════════════════════════════════════════════════════════════
    distThresh = NMS_DIST_FACTOR * meanRadius;
    fprintf('   [4/6] Non-max suppression (dist < %.1f × %d = %.1f)...\n', ...
        NMS_DIST_FACTOR, meanRadius, distThresh);

    keep = true(size(radii));
    for k = 1:numel(radii)
        if ~keep(k), continue; end
        for j = k+1:numel(radii)
            if ~keep(j), continue; end
            d = norm(centers(k,:) - centers(j,:));
            if d < distThresh
                keep(j) = false;
            end
        end
    end
    centers = centers(keep,:);
    radii   = radii(keep);
    fprintf('   After NMS: %d circles\n', numel(radii));

    % ═════════════════════════════════════════════════════════════════
    % STEP 5: Geometry cleanup
    % ═════════════════════════════════════════════════════════════════
    fprintf('   [5/6] Geometry cleanup (edge removal + overlap shrink)...\n');
    numCircles = numel(radii);

    % Edge-cut detection
    isEdge = (centers(:,1) - radii < 1) | ...
             (centers(:,2) - radii < 1) | ...
             (centers(:,1) + radii > W) | ...
             (centers(:,2) + radii > H);

    % Shrink overlapping circles — PROPORTIONAL method
    % Each circle loses only half the overlap amount + 0.5px safety gap
    % Example: overlap=4px → each shrinks by 2.5px (not d/2 which could be huge)
    isShrunk = false(numCircles, 1);
    for i = 1:numCircles
        for j = i+1:numCircles
            d = norm(centers(i,:) - centers(j,:));
            overlap = (radii(i) + radii(j)) - d;
            if overlap > 0
                % Each circle gives up half the overlap + tiny gap
                shrinkAmount = overlap / 2 + 0.5;
                newRi = radii(i) - shrinkAmount;
                newRj = radii(j) - shrinkAmount;
                % Only apply if both radii stay positive and reasonable
                if newRi > 5 && newRj > 5
                    radii(i) = newRi;
                    radii(j) = newRj;
                    isShrunk(i) = true;
                    isShrunk(j) = true;
                end
            end
        end
    end

    nOK      = sum(~isEdge & ~isShrunk);
    nShrunk  = sum(isShrunk & ~isEdge);
    nEdge    = sum(isEdge);
    fprintf('   OK: %d | Shrunk: %d | Edge-cut: %d\n', nOK, nShrunk, nEdge);

    % ═════════════════════════════════════════════════════════════════
    % STEP 6: Consistent numbering (left→right, top→bottom)
    %   Usable circles (green + yellow) numbered FIRST: 1, 2, 3, ...
    %   Edge-cut circles (red) numbered LAST: N+1, N+2, ...
    %   This way cropping goes 1 to N sequentially, no gaps.
    % ═════════════════════════════════════════════════════════════════
    fprintf('   [6/6] Numbering + annotating %d frames...\n', numFrames);

    % Separate usable vs edge indices
    usableIdx = find(~isEdge);
    edgeIdx   = find(isEdge);

    % Sort usable circles left→right, top→bottom
    if ~isempty(usableIdx)
        usableOrder = assignConsistentNumbering(centers(usableIdx, :));
        usableIdx   = usableIdx(usableOrder);
    end

    % Sort edge circles left→right, top→bottom (for consistency)
    if ~isempty(edgeIdx)
        edgeOrder = assignConsistentNumbering(centers(edgeIdx, :));
        edgeIdx   = edgeIdx(edgeOrder);
    end

    % Final order: usable first, then edge
    finalOrder = [usableIdx; edgeIdx];

    % Reorder everything
    centers  = centers(finalOrder, :);
    radii    = radii(finalOrder);
    isEdge   = isEdge(finalOrder);
    isShrunk = isShrunk(finalOrder);
    numbering = (1:numel(radii))';

    nUsable = numel(usableIdx);
    fprintf('   Numbering: 1-%d = usable (green+yellow), %d-%d = edge (red)\n', ...
        nUsable, nUsable+1, numel(radii));

    % ─── Annotate all frames ─────────────────────────────────────────
    for fi = 1:numFrames
        fpath = fullfile(inputPath, frameFiles{fi});
        I = imread(fpath);
        annotated = drawAnnotatedFrame(I, centers, radii, isEdge, isShrunk, ...
                                        numbering, COLOR_OK, COLOR_SHRUNK, COLOR_EDGE, meanRadius);
        imwrite(annotated, fullfile(outImages, frameFiles{fi}));
    end

    % ─── Save logs ───────────────────────────────────────────────────
    fprintf('   Saving logs...\n');
    saveLogs(centers, radii, isEdge, isShrunk, numbering, outLogs, ...
             videoName, numFrames, W, H, meanRadius, ...
             CLAHE_CLIP, MEDIAN_KSIZE, THRESH_VALUE, ...
             PASS1_RAD_RANGE, PASS2_RAD_TOL, NMS_DIST_FACTOR);

    fprintf('   ✅ Done! %d circles: crop 1-%d (usable), %d-%d (edge-cut, skip)\n\n', ...
        numel(radii), nUsable, nUsable+1, numel(radii));
end


%% ─── CONSISTENT NUMBERING ──────────────────────────────────────────────
function order = assignConsistentNumbering(centers)
% Sort circles left→right, top→bottom.
% Groups into rows by Y coordinate, then sorts by X within each row.

    if isempty(centers)
        order = [];
        return;
    end

    n = size(centers, 1);

    % Estimate row threshold from Y spacing
    sortedY = sort(centers(:, 2));
    if n > 1
        yDiffs = diff(sortedY);
        bigGaps = yDiffs(yDiffs > 5);
        if ~isempty(bigGaps)
            rowThreshold = median(bigGaps) * 0.6;
        else
            rowThreshold = 30;
        end
    else
        rowThreshold = 30;
    end

    % Sort by Y first
    [~, yOrder] = sort(centers(:, 2));
    sortedCenters = centers(yOrder, :);

    % Group into rows
    rows = {};
    currentRow = yOrder(1);
    currentY   = sortedCenters(1, 2);

    for i = 2:n
        if abs(sortedCenters(i, 2) - currentY) < rowThreshold
            currentRow = [currentRow; yOrder(i)]; %#ok<AGROW>
        else
            rows{end+1} = currentRow; %#ok<AGROW>
            currentRow = yOrder(i);
            currentY   = sortedCenters(i, 2);
        end
    end
    rows{end+1} = currentRow;

    % Sort each row by X
    order = [];
    for r = 1:numel(rows)
        rowIdx = rows{r};
        [~, xSort] = sort(centers(rowIdx, 1));
        order = [order; rowIdx(xSort)]; %#ok<AGROW>
    end
end


%% ─── ANNOTATED FRAME DRAWING ───────────────────────────────────────────
function annotated = drawAnnotatedFrame(I, centers, radii, isEdge, isShrunk, ...
                                         numbering, COLOR_OK, COLOR_SHRUNK, COLOR_EDGE, meanR)
% Draw colored circles + number labels on one frame.
%   green  = OK
%   yellow = shrunk
%   red    = edge-cut

    annotated = I;
    n = numel(radii);

    % Separate by category for batch insertShape calls
    okIdx   = find(~isEdge & ~isShrunk);
    shIdx   = find(isShrunk & ~isEdge);
    edIdx   = find(isEdge);

    % Draw circles by category
    if ~isempty(okIdx)
        circData = [centers(okIdx,:), radii(okIdx)];
        annotated = insertShape(annotated, 'Circle', circData, ...
            'Color', COLOR_OK, 'LineWidth', 2);
    end
    if ~isempty(shIdx)
        circData = [centers(shIdx,:), radii(shIdx)];
        annotated = insertShape(annotated, 'Circle', circData, ...
            'Color', COLOR_SHRUNK, 'LineWidth', 2);
    end
    if ~isempty(edIdx)
        circData = [centers(edIdx,:), radii(edIdx)];
        annotated = insertShape(annotated, 'Circle', circData, ...
            'Color', COLOR_EDGE, 'LineWidth', 2);
    end

    % Draw center dots (small green circles)
    for i = 1:n
        cx = round(centers(i,1));
        cy = round(centers(i,2));
        annotated = insertShape(annotated, 'FilledCircle', [cx, cy, 2], ...
            'Color', [0 255 0], 'Opacity', 1);
    end

    % Number labels with background
    txtLabels = arrayfun(@(x) sprintf('%d', x), numbering, 'UniformOutput', false);
    txtPos    = centers + [-meanR*0.3, -meanR*0.3];

    % Clamp positions to image bounds
    txtPos(:,1) = max(2, min(txtPos(:,1), size(I,2) - 20));
    txtPos(:,2) = max(12, min(txtPos(:,2), size(I,1) - 2));

    annotated = insertText(annotated, txtPos, txtLabels, ...
        'FontSize', max(10, round(meanR * 0.35)), ...
        'BoxColor', [40 40 40], 'BoxOpacity', 0.7, ...
        'TextColor', 'white');
end


%% ─── SAVE LOGS (CSV, JSON, PLOTS) ─────────────────────────────────────
function saveLogs(centers, radii, isEdge, isShrunk, numbering, logDir, ...
                  videoName, totalFrames, imgW, imgH, meanRadius, ...
                  CLAHE_CLIP, MEDIAN_KSIZE, THRESH_VALUE, ...
                  PASS1_RAD_RANGE, PASS2_RAD_TOL, NMS_DIST_FACTOR)

    n = numel(radii);

    % ─── circle_positions.csv ────────────────────────────────────────
    csvPath = fullfile(logDir, 'circle_positions.csv');
    fid = fopen(csvPath, 'w');
    fprintf(fid, 'circle_id,center_x,center_y,radius,status,is_edge,is_shrunk\n');
    for i = 1:n
        if isEdge(i)
            status = 'edge';
        elseif isShrunk(i)
            status = 'shrunk';
        else
            status = 'ok';
        end
        fprintf(fid, '%d,%.2f,%.2f,%.2f,%s,%s,%s\n', ...
            numbering(i), centers(i,1), centers(i,2), radii(i), ...
            status, string(isEdge(i)), string(isShrunk(i)));
    end
    fclose(fid);

    % ─── detection_metadata.json ─────────────────────────────────────
    nOK     = sum(~isEdge & ~isShrunk);
    nShrunk = sum(isShrunk & ~isEdge);
    nEdge   = sum(isEdge);

    meta = struct();
    meta.video_name              = videoName;
    meta.total_circles           = n;
    meta.ok_circles              = nOK;
    meta.shrunk_circles          = nShrunk;
    meta.edge_cut_circles        = nEdge;
    meta.usable_circles          = sum(~isEdge);
    meta.crop_range_start        = 1;
    meta.crop_range_end          = sum(~isEdge);
    meta.edge_range_start        = sum(~isEdge) + 1;
    meta.edge_range_end          = n;
    meta.numbering_note          = 'Circles 1 to crop_range_end are usable (green+yellow). Circles after that are edge-cut (red). Crop sequentially 1 to crop_range_end, skip the rest.';
    meta.estimated_mean_radius   = round(meanRadius * 100) / 100;
    meta.total_frames_annotated  = totalFrames;
    meta.image_resolution        = sprintf('%dx%d', imgW, imgH);

    meta.detection_parameters.clahe_clip          = CLAHE_CLIP;
    meta.detection_parameters.median_ksize        = MEDIAN_KSIZE;
    meta.detection_parameters.threshold           = THRESH_VALUE;
    meta.detection_parameters.pass1_radius_range  = PASS1_RAD_RANGE;
    meta.detection_parameters.pass2_radius_tolerance = PASS2_RAD_TOL;
    meta.detection_parameters.hough_sensitivity   = 1.0;
    meta.detection_parameters.hough_edge_threshold = 0.01;
    meta.detection_parameters.nms_dist_factor     = NMS_DIST_FACTOR;

    if n > 0
        meta.radius_stats.min_val    = round(min(radii) * 100) / 100;
        meta.radius_stats.max_val    = round(max(radii) * 100) / 100;
        meta.radius_stats.mean_val   = round(mean(radii) * 100) / 100;
        meta.radius_stats.median_val = round(median(radii) * 100) / 100;
    else
        meta.radius_stats.min_val    = 0;
        meta.radius_stats.max_val    = 0;
        meta.radius_stats.mean_val   = 0;
        meta.radius_stats.median_val = 0;
    end

    meta.processed_at = datestr(now, 'yyyy-mm-ddTHH:MM:SS');

    jsonText = jsonencode(meta);
    % Pretty-print: add newlines after commas/braces for readability
    jsonText = strrep(jsonText, ',"', sprintf(',\n  "'));
    jsonText = strrep(jsonText, '{', sprintf('{\n  '));
    jsonText = strrep(jsonText, '}', sprintf('\n}'));

    fid = fopen(fullfile(logDir, 'detection_metadata.json'), 'w');
    fprintf(fid, '%s', jsonText);
    fclose(fid);

    % ─── radius_distribution.png ─────────────────────────────────────
    if n > 0
        fig = figure('Visible', 'off', 'Position', [100 100 800 400]);
        [sortedR, sortIdx] = sort(radii);
        barColors = zeros(n, 3);
        for i = 1:n
            origI = sortIdx(i);
            if isEdge(origI)
                barColors(i,:) = [0.86 0.08 0.24];  % red
            elseif isShrunk(origI)
                barColors(i,:) = [1.0  0.84 0.0 ];  % gold
            else
                barColors(i,:) = [0.13 0.55 0.13];  % green
            end
        end
        b = bar(sortedR, 'FaceColor', 'flat');
        b.CData = barColors;
        hold on;
        yline(meanRadius, '--b', sprintf('Mean: %.1fpx', meanRadius), ...
            'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
        xlabel('Circle (sorted by radius)');
        ylabel('Radius (pixels)');
        title(sprintf('Circle Radii — %s', videoName), 'Interpreter', 'none');
        grid on; grid minor;
        set(gca, 'GridAlpha', 0.3);
        saveas(fig, fullfile(logDir, 'radius_distribution.png'));
        close(fig);
    end

    % ─── circle_map.png ──────────────────────────────────────────────
    if n > 0
        fig = figure('Visible', 'off', 'Position', [100 100 1200 700]);
        hold on;
        axis equal;
        set(gca, 'YDir', 'reverse');
        xlim([0 imgW]);
        ylim([0 imgH]);

        for i = 1:n
            if isEdge(i)
                ec = [0.86 0.08 0.24];
            elseif isShrunk(i)
                ec = [1.0  0.84 0.0 ];
            else
                ec = [0.13 0.55 0.13];
            end
            theta = linspace(0, 2*pi, 100);
            cx = centers(i,1) + radii(i) * cos(theta);
            cy = centers(i,2) + radii(i) * sin(theta);
            plot(cx, cy, 'Color', ec, 'LineWidth', 0.8);
            text(centers(i,1), centers(i,2), sprintf('%d', numbering(i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                'FontSize', 6, 'Color', [0 0 0.8]);
        end

        title(sprintf('Circle Map — %s (green=%d, yellow=%d, red=%d)', ...
            videoName, nOK, nShrunk, nEdge), 'Interpreter', 'none');
        xlabel('X (pixels)');
        ylabel('Y (pixels)');
        saveas(fig, fullfile(logDir, 'circle_map.png'));
        close(fig);
    end
end


%% ─── NATURAL SORT HELPER ───────────────────────────────────────────────
function sorted = natsortfiles(fileList)
% Simple natural sort for filenames like frame_001.png, frame_002.png, etc.
% Extracts numeric part and sorts by that.
    nums = zeros(numel(fileList), 1);
    for i = 1:numel(fileList)
        tokens = regexp(fileList{i}, '(\d+)', 'tokens');
        if ~isempty(tokens)
            nums(i) = str2double(tokens{end}{1});
        else
            nums(i) = i;
        end
    end
    [~, idx] = sort(nums);
    sorted = fileList(idx);
end