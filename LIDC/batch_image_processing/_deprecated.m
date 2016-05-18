% f = fopen(strcat(sampleInputFolderRoot, '\', sampleEdgeVerticesFileName));
% m = [];
% while (~feof(f))
%     a = textscan(f, '%s,%s,%f', 'delimiter', ',');
% %     strcat('',a{1}, ' | ', a{2}, ' | ', num2str(a{3}))
% %     str2double(a{1})
% %     celldisp(a);
%     c1 = str2double(a{1});
%     c2 = str2double(a{2});
%     vertcat(m, [c1, c2, a{3});
% 
% %     a = fgetl(f);
% %     disp(a);
% 
% 
% end
% fclose(f);


% sampleEdgeVertices = readtable(sampleInputFilePath, 'Delimiter',',', 'ReadVariableNames', true, 'Format', '%s%s%u');
% for i=1 : size(sampleEdgeVertices, 1)
%     sampleEdgeVertices(i,:).x___xCoord = str2double(sampleEdgeVertices(i,:).x___xCoord);
%     sampleEdgeVertices(i,:).yCoord = str2double(sampleEdgeVertices(i,:).yCoord);
% end



% sampleEdgeVertices = [
%     [312, 355]
%     [311, 356]
%     [310, 357]
%     [309, 357]
%     [308, 358]
%     [308, 359]
%     [308, 360]
%     [307, 360]
%     [306, 361]
%     [306, 362]
%     [305, 363]
%     [304, 364]
%     [303, 365]
%     [303, 366]
%     [302, 367]
%     [302, 368]
%     [302, 369]
%     [301, 370]
%     [301, 371]
%     [300, 371]
%     [299, 372]
%     [299, 373]
%     [299, 374]
%     [299, 375]
%     [299, 376]
%     [300, 377]
%     [301, 378]
%     [302, 379]
%     [303, 379]
%     [304, 379]
%     [305, 379]
%     [306, 379]
%     [307, 378]
%     [308, 377]
%     [308, 376]
%     [309, 375]
%     [310, 375]
%     [311, 375]
%     [312, 375]
%     [313, 375]
%     [314, 375]
%     [315, 375]
%     [316, 375]
%     [317, 375]
%     [318, 375]
%     [319, 375]
%     [320, 374]
%     [321, 373]
%     [322, 372]
%     [322, 371]
%     [322, 370]
%     [323, 369]
%     [324, 368]
%     [325, 367]
%     [326, 366]
%     [327, 365]
%     [328, 364]
%     [328, 363]
%     [327, 362]
%     [327, 361]
%     [326, 360]
%     [325, 359]
%     [324, 359]
%     [323, 358]
%     [322, 358]
%     [321, 357]
%     [320, 358]
%     [319, 358]
%     [318, 358]
%     [318, 357]
%     [317, 356]
%     [316, 355]
%     [315, 355]
%     [314, 355]
%     [313, 355]
%     [312, 355]
% ];



% % Read the ROI file.
% [imageZposition,imageSOP_UID,inclusion,roi_Id,unblindedReadNodule_Id] = importRoi(strcat(sampleInputFolderRoot, '\', sampleRoiFileName));
% % remove the double quotes, if they exist.
% for i=1 : size(imageSOP_UID, 1)
%     s = imageSOP_UID(i);
%     imageSOP_UID(i) = strrep(imageSOP_UID(i), '"', '');
% end
% % Read the ROI edges file.
% [roiPtX, roiPtY, roiId] = importRoiEdges(strcat(sampleInputFolderRoot, '\', sampleEdgeVerticesFileName));
% allEdges = [roiPtX, roiPtY, roiId];
% 
% 
% % 0) For each ROI:
% for roiId = 0 : size(roi_Id, 1) + 1
%     sampleEdgeVertices = allEdges(find(allEdges(:,3) == roiId), :);
% 
%     % STEP 1: Find the file to which the ROI edges apply.
%     selectedRoiIdx = find(roi_Id == roiId);
%     sampleSopUID = imageSOP_UID(selectedRoiIdx);
%     [fp, fileName] = findLidcXmlImageFileByTargetSOPInstanceUID(sampleInputFolderRoot, sampleSopUID);
%     disp(strcat(num2str(roiId), ':', sampleSopUID, ': Input image file is: ', fp));
% 
%     % 2) read original scan into memory
%     [dicomImg, map] = dicomread(fp);
% 
%     % 3) convert original scan to RGB space
%     grayImg = mat2gray(dicomImg);
%     rgbImg = reshape([grayImg grayImg grayImg], [size(grayImg) 3]);
% 
%     % 4) generate the new set of extracts for each scan.
%     generateImages(rgbImg, roiId, sampleEdgeVertices, sampleSopUID, sampleOutputFolder, fileName);
% 
% end