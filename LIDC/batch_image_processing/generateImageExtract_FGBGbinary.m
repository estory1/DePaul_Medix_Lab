% generateImageExtract_FGBGbinary.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Generates the foreground, background, and binary image extracts, for the given RGB image,
function [fFG, fBG, fBinary] = generateImageExtract_FGBGbinary(rgbImg, outputFolder, fileName, sopUid, csvData, roisForImgSopUID)

    % Define a prefix for the output file name.
    outputFileNamePrefix = strrep(fileName, '.dcm', '-');
  
    % PK, part 1: Extract the XMLStudyNode so we know where to look in the edgeMap file.
    xmlStudyNode = regexprep(outputFolder, '.*(LIDC-IDRI-\d\d\d\d).*', '$1');
    % PK, part 2: Extract the FileNode, so know where to look in the edgeMap file.
    fileNodeCandidateIdxs = find(strcmp(csvData.edges{4}, xmlStudyNode));
    fileNode = max(csvData.edges{7}(fileNodeCandidateIdxs));

    % Get the sorted set of ROIs, with associated information converted into numeric form and copied into a matrix we can easily sort.
    sortedRoiMat = generateImageExtracts_getSortedRoisForNodule(csvData, roisForImgSopUID, outputFolder);

    % Get just the ROIs defining an area of inclusion. (Typically, the
    % ROI's outer contour.)
    sortedRoiMatInclusionsOnly = sortedRoiMat(find(sortedRoiMat(:,3) == 1),:);
    
    % For each of the inclusion ROIs...
    for i=1 : size(sortedRoiMatInclusionsOnly, 1)
        
        % Look-up the values that can't be stored into a cell array and
        % then sorted...
        roiId = sortedRoiMatInclusionsOnly(i,1);
%         inclusionNum = sortedRoiMatInclusionsOnly(i,3);
        rsId = sortedRoiMatInclusionsOnly(i,8);
%         inclusionStr = 'FALSE'; if inclusionNum == 1; inclusionStr = 'TRUE'; end

        % Get the vertices for this ROI.
        roiVerticesIdxs = find(csvData.edges{3} == roiId & strcmp(csvData.edges{4}, xmlStudyNode) & csvData.edges{7} == fileNode);
        roiVertices = [csvData.edges{1}(roiVerticesIdxs), csvData.edges{2}(roiVerticesIdxs)];
        % Alias the vertex coords to simpler variables...
        x = roiVertices(:,1);
        y = roiVertices(:,2);

        % Generate extract; rectangle-bounded ROI overlay on scan.
        [mnmn, mnmx, mxmn, mxmx] = computeMinBoundingRectangle(roiVertices);
        bbox = [mnmn(1), mnmn(2),  mxmx(1) - mnmn(1), mxmx(2) - mnmn(2)];

        
        %% Generate binary image.
        bin = zeros(size(rgbImg, 1), size(rgbImg, 2));  % black background
        bin = roipoly(bin, x, y);                       % draw a white ROI
        
        % Apply the exclusion region, if exists, via:
        %   1) Search for an exclusion region outlined during the current ROI's reading session.
        sortedRoiMatExclusionsOnlyForRadiologist = sortedRoiMat(find(sortedRoiMat(:,3) == 0 & sortedRoiMat(:,8) == rsId));

        % Binary image to contain the exclusion ROI.
        binExcl = zeros(size(rgbImg, 1), size(rgbImg, 2));  % black background

        % For each exclusion ROI...
        for j=1 : size(sortedRoiMatExclusionsOnlyForRadiologist, 1)
            %   2) Draw the exclusion region in a buffer image.
            roiExcl = sortedRoiMat(j,1);
            roiVerticesIdxsExcl = find (csvData.edges{3} == roiExcl & strcmp(csvData.edges{4}, xmlStudyNode) & csvData.edges{7} == fileNode);
            roiVerticesExcl = [csvData.edges{1}(roiVerticesIdxsExcl), csvData.edges{2}(roiVerticesIdxsExcl)];
            xExcl = roiVerticesExcl(:,1);
            yExcl = roiVerticesExcl(:,2);
            binExcl = roipoly(binExcl, xExcl, yExcl);
            %   3) Invert the exclusion region.
            binExcl = ~binExcl;
            %   4) Multiply the main binary image by this image.
            bin = binExcl .* bin;
        end

        % Save the binary image to disk.
        binPostCrop = imcrop(bin, bbox);                        % crop to the ROI
        
        imshow(binPostCrop);
        binFilePath =  strcat( outputFolder, filesep, outputFileNamePrefix, 'roi_', num2str(roiId), '-rs_', num2str(rsId), '-', 'bin.tiff' );
        imwrite(binPostCrop, binFilePath);

        
        %% Generate the crop iamge.
        crop = rgb2gray(rgbImg);
        crop =imcrop(crop, bbox);
        imshow(crop);
        cropFilePath =  strcat( outputFolder, filesep, outputFileNamePrefix, 'roi_', num2str(roiId), '-rs_', num2str(rsId), '-', 'crop.tiff' );
        imwrite(crop, cropFilePath);

        %% Generate the foreground image.
        fg = bin .* crop;
        imshow(fg);
        fgFilePath =  strcat( outputFolder, filesep, outputFileNamePrefix, 'roi_', num2str(roiId), '-rs_', num2str(rsId), '-', 'fg.tiff' );
        imwrite(fg, fgFilePath);

        %% Generate the background image.
        bg = ~bin .* crop;
        imshow(bg);
        bgFilePath =  strcat( outputFolder, filesep, outputFileNamePrefix, 'roi_', num2str(roiId), '-rs_', num2str(rsId), '-', 'bg.tiff' );
        imwrite(bg, bgFilePath);
        
        
        %% Generate the patches (subsets): original image.
        patchRoot = fullfile(strcat(outputFolder, filesep), 'patches', filesep, 'orig');
        mkdir(patchRoot);
        numCreated = genPatchSet(patchRoot, fileName, rgbImg);
        
        %% Generate the patches (subsets): binary.
        patchRoot = fullfile(strcat(outputFolder, filesep), 'patches', filesep, 'bin');
        mkdir(patchRoot);
        numCreated = genPatchSet(patchRoot, fileName, bin);
          
        %% Generate the patches.
        numCreated = genPatchSet(outputFolder, fileName, roiId, rsId, rgbImg, bin);

    end
        
end