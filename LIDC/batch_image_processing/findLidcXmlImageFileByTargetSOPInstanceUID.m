% findLidcXmlImageFileByTargetSOPInstanceUID.m
% 
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Searches the LIDC image set from a specified folder root and 
%% finds the 1:1 map from the SOPInstanceUID to its DICOM image file,
%% returning the DICOM file path. EXTREMELY USEFUL!
function [fp, fileName] = findLidcXmlImageFileByTargetSOPInstanceUID(folderRoot, targetSOPInstanceUID)

    % STEP 1: Find the file to which the ROI edges apply.
    fileNames = dir(strcat(fullfile(folderRoot), filesep, '*.dcm'));
    for i = 1 : length(fileNames)
        fileName = fileNames(i).name;
        fp = fullfile(folderRoot, fileName);
        di = dicominfo(fp);
        if strcmp(di.SOPInstanceUID, targetSOPInstanceUID) == 1
    %        disp(strcat('Target file is: ', fp));
           break;
        end
    end
    
end