% generateImageExtractsForScan.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Generates images for the given RGB image and ROI vertices (inaccurately named "edges").
function x = generateImageExtractsForScan(rgbImg, outputFolder, fileName, sopUid, csvData, roisForImgSopUID, dcmData, studyId, seriesId)

    % Generate a full size image showing each radiologist's ROI as a diff. color.
    generateAndSaveImageExtract_RoiBounded(rgbImg, outputFolder, fileName, sopUid, csvData, roisForImgSopUID, dcmData, studyId, seriesId);

    % Generate binary, FB, and BG crops.
    generateImageExtract_FGBGbinary(rgbImg, outputFolder, fileName, sopUid, csvData, roisForImgSopUID);

end