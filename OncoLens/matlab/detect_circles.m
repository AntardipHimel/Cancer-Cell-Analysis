function detect_circles(input_dir, output_dir, video_name, save_images)
%% DETECT_CIRCLES - Circle Detection for OncoLens
%
% Detects circular lenses in microscopy images using Hough Transform.
% Matches the original 1_3_detect_circles.m pipeline EXACTLY.
%
% USAGE:
%   detect_circles('path/to/frames', 'path/to/output', 'video_name')
%   detect_circles('path/to/frames', 'path/to/output', 'video_name', true)
%   detect_circles('path/to/frames', 'path/to/output', 'video_name', false)
%
% INPUTS:
%   input_dir   - Directory containing extracted frames (frame_001.png, etc.)
%   output_dir  - Directory for output files
%   video_name  - Name of the video being processed
%   save_images - (Optional) true = save annotated frames, false = data only
%                 Default: true
%
% PIPELINE:
%   1. Load frame_001.png → adapthisteq → medfilt2 → threshold (binary mask)
%   2. Pass 1: wide Hough [30,50] to estimate mean radius
%   3. Pass 2: tight Hough [meanR-5, meanR+5] on binary mask
%   4. Non-max suppression to remove duplicates
%   5. Geometry cleanup: remove edge-cut circles, shrink overlapping ones
%   6. Consistent numbering (left→right, top→bottom)
%      Usable circles (green + yellow) numbered FIRST: 1, 2, 3, ...
%      Edge-cut circles (red) numbered LAST: N+1, N+2, ...
%   7. Annotate ALL 30 frames with colored circles:
%      green = OK, yellow = shrunk, red = edge-removed
%   8. Save logs (CSV, JSON, plots)
%
% OUTPUTS:
%   output_dir/images/          - Annotated frames (if save_images=true)
%   output_dir/logs/
%       ├── circle_positions.csv
%       ├── detection_metadata.json
%       ├── qc_01_enhanced.png
%       ├── qc_02_smoothed.png
%       ├── qc_03_binary_mask.png
%       ├── radius_distribution.png
%       └── circle_map.png
%
% Author: Antardip Himel
% Date: March 2026
% Part of OncoLens - Cancer Cell Classification Software

%% ═══════════════════════════════════════════════════════════════════════
%  PARAMETERS - EDIT THESE TO TUNE DETECTION
%  ═══════════════════════════════════════════════════════════════════════

% Preprocessing
CLAHE_CLIP   = 0.02;        % adapthisteq ClipLimit
CLAHE_TILES  = [8 8];       % adapthisteq NumTiles
MEDIAN_KSIZE = [5 5];       % medfilt2 kernel
THRESH_VALUE = 200;         % Binary threshold (S0 > 200)

% Hough — Pass 1 (wide radius estimation)
PASS1_RAD_RANGE = [30 50];

% Hough — Pass 2 (tight around mean)
PASS2_RAD_TOL = 5;

% Hough tuning
HOUGH_SENSITIVITY  = 1.0;
HOUGH_EDGE_THRESH  = 0.01;   % tight pass uses lower edge threshold
HOUGH_METHOD       = 'TwoStage';
HOUGH_POLARITY     = 'dark';

% NMS
NMS_DIST_FACTOR = 1.6;

% Annotation colors (RGB 0-255 for insertShape)
COLOR_OK     = [0   200 0  ];   % Green
COLOR_SHRUNK = [220 220 0  ];   % Yellow
COLOR_EDGE   = [220 0   0  ];   % Red

%% ═══════════════════════════════════════════════════════════════════════
%  INITIALIZATION
%  ═══════════════════════════════════════════════════════════════════════

% Handle optional save_images argument
if nargin < 4
    save_images = true;
end
if ischar(save_images) || isstring(save_images)
    save_images = strcmpi(save_images, 'true') || strcmp(save_images, '1');
end

clc;
fprintf('═══════════════════════════════════════════════════════════════════\n');
fprintf('  ONCOLENS - CIRCLE DETECTION\n');
fprintf('  %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('═══════════════════════════════════════════════════════════════════\n\n');

fprintf('  Input:  %s\n', input_dir);
fprintf('  Output: %s\n', output_dir);
fprintf('  Video:  %s\n', video_name);
fprintf('  Save annotated images: %s\n\n', string(save_images));

% Validate input directory
if ~exist(input_dir, 'dir')
    error('Input directory does not exist: %s', input_dir);
end

% Create output directories
outImages = fullfile(output_dir, 'images');
outLogs   = fullfile(output_dir, 'logs');

if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if save_images && ~exist(outImages, 'dir'), mkdir(outImages); end
if ~exist(outLogs, 'dir'), mkdir(outLogs); end

%% ═══════════════════════════════════════════════════════════════════════
%  FIND AND LOAD FRAMES
%  ═══════════════════════════════════════════════════════════════════════

fprintf('───────────────────────────────────────────────────────────────────\n');
fprintf('  Loading frames...\n');

% Find frame files (frame_001.png format)
frameFiles = dir(fullfile(input_dir, 'frame_*.png'));
if isempty(frameFiles)
    frameFiles = dir(fullfile(input_dir, 'frame_*.jpg'));
end
if isempty(frameFiles)
    % Also try generic image files
    frameFiles = dir(fullfile(input_dir, '*.png'));
    if isempty(frameFiles)
        frameFiles = dir(fullfile(input_dir, '*.jpg'));
    end
end
if isempty(frameFiles)
    error('No image files found in: %s', input_dir);
end

% Natural sort (using local helper to avoid collision with File Exchange natsortfiles)
frameFiles = oncolens_natsort({frameFiles.name});
numFrames  = numel(frameFiles);

% Load first frame
firstPath = fullfile(input_dir, frameFiles{1});
I0 = imread(firstPath);
[H, W, nChannels] = size(I0);

if nChannels == 3
    G0 = rgb2gray(I0);
else
    G0 = I0;
end

fprintf('  Found %d frames (%dx%d)\n', numFrames, W, H);
fprintf('  First frame: %s\n\n', frameFiles{1});

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 1: PREPROCESS → BINARY MASK
%  ═══════════════════════════════════════════════════════════════════════

fprintf('  [1/6] Preprocessing → binary mask (threshold=%d)...\n', THRESH_VALUE);

% CLAHE enhancement
E0 = adapthisteq(G0, 'ClipLimit', CLAHE_CLIP, 'NumTiles', CLAHE_TILES);

% Median filter
S0 = medfilt2(E0, MEDIAN_KSIZE);

% Binary threshold
mask0 = uint8(S0 > THRESH_VALUE) * 255;

% Save QC images
imwrite(E0,    fullfile(outLogs, 'qc_01_enhanced.png'));
imwrite(S0,    fullfile(outLogs, 'qc_02_smoothed.png'));
imwrite(mask0, fullfile(outLogs, 'qc_03_binary_mask.png'));

fprintf('        Saved QC images to logs/\n');

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 2: PASS 1 — ESTIMATE MEAN RADIUS
%  ═══════════════════════════════════════════════════════════════════════

fprintf('  [2/6] Pass 1: estimating mean radius [r=%d-%d]...\n', ...
    PASS1_RAD_RANGE(1), PASS1_RAD_RANGE(2));

[~, r0] = imfindcircles(mask0, PASS1_RAD_RANGE, ...
    'ObjectPolarity', HOUGH_POLARITY, ...
    'Sensitivity',    HOUGH_SENSITIVITY, ...
    'EdgeThreshold',  0.1, ...
    'Method',         HOUGH_METHOD);

if isempty(r0)
    error('No circles found in pass 1. Try adjusting PASS1_RAD_RANGE.');
end

meanRadius = round(mean(r0));
fprintf('        Pass 1: %d circles, mean radius = %dpx\n', numel(r0), meanRadius);

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 3: PASS 2 — TIGHT DETECTION
%  ═══════════════════════════════════════════════════════════════════════

rMin = max(10, meanRadius - PASS2_RAD_TOL);
rMax = meanRadius + PASS2_RAD_TOL;
fprintf('  [3/6] Pass 2: tight detection [r=%d-%d]...\n', rMin, rMax);

[centers, radii] = imfindcircles(mask0, [rMin rMax], ...
    'ObjectPolarity', HOUGH_POLARITY, ...
    'Sensitivity',    HOUGH_SENSITIVITY, ...
    'EdgeThreshold',  HOUGH_EDGE_THRESH, ...
    'Method',         HOUGH_METHOD);

% Fallback: wider range if no circles found
if isempty(centers)
    fprintf('        ⚠ No circles in pass 2 — trying wider range...\n');
    rMin2 = max(10, meanRadius - 10);
    rMax2 = meanRadius + 10;
    [centers, radii] = imfindcircles(mask0, [rMin2 rMax2], ...
        'ObjectPolarity', HOUGH_POLARITY, ...
        'Sensitivity',    HOUGH_SENSITIVITY, ...
        'EdgeThreshold',  HOUGH_EDGE_THRESH, ...
        'Method',         HOUGH_METHOD);
    if isempty(centers)
        error('No circles found even with wider range. Check preprocessing.');
    end
end

fprintf('        Pass 2: %d circles detected\n', size(centers, 1));

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 4: NON-MAX SUPPRESSION
%  ═══════════════════════════════════════════════════════════════════════

distThresh = NMS_DIST_FACTOR * meanRadius;
fprintf('  [4/6] Non-max suppression (dist < %.1f × %d = %.1f)...\n', ...
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
fprintf('        After NMS: %d circles\n', numel(radii));

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 5: GEOMETRY CLEANUP
%  ═══════════════════════════════════════════════════════════════════════

fprintf('  [5/6] Geometry cleanup (edge removal + overlap shrink)...\n');
numCircles = numel(radii);

% ─── Edge-cut detection ───────────────────────────────────────────────
% Circle is edge-cut if it extends beyond image boundaries
isEdge = (centers(:,1) - radii < 1) | ...
         (centers(:,2) - radii < 1) | ...
         (centers(:,1) + radii > W) | ...
         (centers(:,2) + radii > H);

% ─── Shrink overlapping circles ───────────────────────────────────────
% PROPORTIONAL method: Each circle loses half the overlap + 0.5px safety gap
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

nOK     = sum(~isEdge & ~isShrunk);
nShrunk = sum(isShrunk & ~isEdge);
nEdge   = sum(isEdge);
fprintf('        OK: %d | Shrunk: %d | Edge-cut: %d\n', nOK, nShrunk, nEdge);

%% ═══════════════════════════════════════════════════════════════════════
%  STEP 6: CONSISTENT NUMBERING
%  ═══════════════════════════════════════════════════════════════════════
%
%  Usable circles (green + yellow) numbered FIRST: 1, 2, 3, ...
%  Edge-cut circles (red) numbered LAST: N+1, N+2, ...
%  This way cropping goes 1 to N sequentially, no gaps.
%
%  Within each group, sort left→right, top→bottom.
%  ═══════════════════════════════════════════════════════════════════════

fprintf('  [6/6] Numbering + annotating %d frames...\n', numFrames);

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
fprintf('        Numbering: 1-%d = usable (green+yellow), %d-%d = edge (red)\n', ...
    nUsable, nUsable+1, numel(radii));

%% ═══════════════════════════════════════════════════════════════════════
%  ANNOTATE ALL FRAMES
%  ═══════════════════════════════════════════════════════════════════════

if save_images
    fprintf('        Annotating %d frames...\n', numFrames);
    for fi = 1:numFrames
        fpath = fullfile(input_dir, frameFiles{fi});
        I = imread(fpath);
        annotated = drawAnnotatedFrame(I, centers, radii, isEdge, isShrunk, ...
                                        numbering, COLOR_OK, COLOR_SHRUNK, COLOR_EDGE, meanRadius);
        imwrite(annotated, fullfile(outImages, frameFiles{fi}));
    end
    fprintf('        Saved %d annotated frames to images/\n', numFrames);
else
    fprintf('        Skipping image annotation (save_images=false)\n');
end

%% ═══════════════════════════════════════════════════════════════════════
%  SAVE LOGS
%  ═══════════════════════════════════════════════════════════════════════

fprintf('        Saving logs...\n');
saveLogs(centers, radii, isEdge, isShrunk, numbering, outLogs, ...
         video_name, numFrames, W, H, meanRadius, ...
         CLAHE_CLIP, MEDIAN_KSIZE, THRESH_VALUE, ...
         PASS1_RAD_RANGE, PASS2_RAD_TOL, NMS_DIST_FACTOR);

%% ═══════════════════════════════════════════════════════════════════════
%  SUMMARY
%  ═══════════════════════════════════════════════════════════════════════

fprintf('\n═══════════════════════════════════════════════════════════════════\n');
fprintf('  ✅ DETECTION COMPLETE\n');
fprintf('═══════════════════════════════════════════════════════════════════\n\n');
fprintf('  Total circles: %d\n', numel(radii));
fprintf('    🟢 OK:       %d\n', nOK);
fprintf('    🟡 Shrunk:   %d\n', nShrunk);
fprintf('    🔴 Edge-cut: %d\n', nEdge);
fprintf('\n');
fprintf('  Crop range: 1 to %d (usable circles)\n', nUsable);
fprintf('  Skip range: %d to %d (edge-cut circles)\n', nUsable+1, numel(radii));
fprintf('\n');
fprintf('  Output:\n');
fprintf('    logs/circle_positions.csv\n');
fprintf('    logs/detection_metadata.json\n');
if save_images
    fprintf('    images/ (%d annotated frames)\n', numFrames);
end
fprintf('\n');

end


%% ═══════════════════════════════════════════════════════════════════════
%  HELPER FUNCTIONS
%  ═══════════════════════════════════════════════════════════════════════


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

    meta.detection_parameters.clahe_clip             = CLAHE_CLIP;
    meta.detection_parameters.median_ksize           = MEDIAN_KSIZE;
    meta.detection_parameters.threshold              = THRESH_VALUE;
    meta.detection_parameters.pass1_radius_range     = PASS1_RAD_RANGE;
    meta.detection_parameters.pass2_radius_tolerance = PASS2_RAD_TOL;
    meta.detection_parameters.hough_sensitivity      = 1.0;
    meta.detection_parameters.hough_edge_threshold   = 0.01;
    meta.detection_parameters.nms_dist_factor        = NMS_DIST_FACTOR;

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
function sorted = oncolens_natsort(fileList)
% Simple natural sort for filenames like frame_001.png, frame_002.png, etc.
% Extracts numeric part and sorts by that.
% Named uniquely to avoid collision with File Exchange natsortfiles.
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