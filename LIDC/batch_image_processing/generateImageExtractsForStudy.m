% generateImageExtractsForStudy.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%   Handle a patient's CT images at the study level.
%
%%   For an LIDC patient's study, this function loops over all the ROIs and produces a set of extract images (e.g. crops).
function r = generateImageExtractsForStudy(inputFolderRoot, outputFolder, csvData, imageSOP_UID_forThisFolder)

    %% Get values for this folder.
    % Extract the XMLStudyNode so we know where to look in the edgeMap file.
    xmlStudyNode = regexprep(outputFolder, '.*(LIDC-IDRI-\d\d\d\d).*', '$1');
    fileNodeCandidateIdxs = find(strcmp(csvData.edges{4}, xmlStudyNode));
    fileNode = max(csvData.edges{7}(fileNodeCandidateIdxs));
    
    studyId = regexprep(outputFolder, strcat('^.*',  '\', filesep, '(.*?)', '\', filesep, '(.*)$'), '$1');
    seriesId = regexprep(outputFolder, strcat('^.*', '\', filesep, '(.*)$'), '$1');
    
    % 20150301, estory: LIDC-IDRI-0306: Apparently some patients have no
    % ROIs. Prevent further processing of ROIs for these cases.
    if ~isempty(fileNode)
    
        % Get the imageSOP_UID values for just this folder.
        imageSOP_UID_forThisFolder = csvData.roi{2}(find(ismember(csvData.roi{6}, xmlStudyNode) & ismember(csvData.roi{7}, studyId) & ismember(csvData.roi{8}, seriesId) & csvData.roi{9} == fileNode));
        roi_Id_forThisFolder = csvData.roi{4}(find(ismember(csvData.roi{6}, xmlStudyNode) & ismember(csvData.roi{7}, studyId) & ismember(csvData.roi{8}, seriesId) & csvData.roi{9} == fileNode));


        %% Determine the distinct set of CT images for this patient.
        imageSOP_UID_unique = unique(imageSOP_UID_forThisFolder);

        %% Process each CT image.
        for uidIdx=1 : size(imageSOP_UID_unique,1)
            uid = imageSOP_UID_unique(uidIdx);

            % STEP 1: Find the file to which the ROI edges apply.
            [fp, fileName] = findLidcXmlImageFileByTargetSOPInstanceUID(inputFolderRoot, uid);
            disp(strcat('* File path: ', fp));
            disp(strcat('SOP UID: ', uid));

            roisForImgSopUID = roi_Id_forThisFolder(find(ismember(imageSOP_UID_forThisFolder, uid)));

            % 2) read original scan into memory
            [dicomImg, map] = dicomread(fp);
            dcmData.info = dicominfo(fp);
            dcmData.img = dicomImg;
            dcmData.map = map;
            dcmData.fullPath = fp;

            % 3) convert original scan to RGB space
            grayImg = mat2gray(dicomImg);
            rgbImg = reshape([grayImg grayImg grayImg], [size(grayImg) 3]);

            % 4) generate the new set of extracts for each scan.
            generateImageExtractsForScan(rgbImg, outputFolder, fileName, uid, csvData, roisForImgSopUID, dcmData, studyId, seriesId);

            %% Deallocate memory.
            clear dicomImg;
            clear map;
            clear roisForImgSopUID;
            clear grayImg;
            clear rgbImg;
            clear dcmData;
        end
    end
    
end

