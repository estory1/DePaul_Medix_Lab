% generateImageExtracts_getSortedRoisForNodule.m
%
% Original author: Evan Story (estory0@gmail.com)
%
% Purpose:
%
%% Generates a sorted cell matrix of ROIs for a particular nodule.
function sortedRoiCellMat = generateImageExtracts_getSortedRoisForNodule(csvData, roisForImgSopUID, outputFolder)
    
    % Get the ROI count.
    nRows = size(roisForImgSopUID, 1);

    % PK, part 1: Extract the XMLStudyNode so we know where to look in the edgeMap file.
    xmlStudyNode = regexprep(outputFolder, '.*(LIDC-IDRI-\d\d\d\d).*', '$1');
    % PK, part 2: Extract the FileNode, so know where to look in the edgeMap file.
    fileNodeCandidateIdxs = find(strcmp(csvData.edges{4}, xmlStudyNode));
    fileNode = max(csvData.edges{7}(fileNodeCandidateIdxs));

    % Create a cell matrix for the ROIs.
    roiCellMat = cell(nRows, 8);

    % For each radiologist's ROI for this particular image...
    for i=1 : nRows
        roiId = roisForImgSopUID(i);
        
        % Extract the useful facts from our CSV data.
        roiIdx = find(csvData.roi{4} == roiId & strcmp(csvData.roi{6}, xmlStudyNode) & csvData.roi{9} == fileNode);
        inclusion = csvData.roi{3}(roiIdx);
        unId = csvData.roi{5}(roiIdx);
        urnIdx = find(csvData.unblindedReadNodule{2} == unId & strcmp(csvData.unblindedReadNodule{4}, xmlStudyNode) & csvData.unblindedReadNodule{7} == fileNode);
        rsId_urn = csvData.unblindedReadNodule{3}(urnIdx);
        rsIdx = find(csvData.readingSession{3} == rsId_urn & strcmp(csvData.readingSession{5}, xmlStudyNode) & csvData.readingSession{8} == fileNode);
        rsId = csvData.readingSession{3}(rsIdx);
        servicingRadiologistID = csvData.readingSession{2}(rsIdx);
        % Convert the radiologist ID to numeric if it's not already.
        if isnumeric(servicingRadiologistID); servicingRadiologistID = num2str(servicingRadiologistID); end
        
        % Get the inclusion value as numeric. Also, this is a hack around MATLAB's seriously terrible CSV file reading.
        inclusionNum = 0;
        for j=1 : length(inclusion)
            if strcmpi(inclusion(j), 'TRUE')
                inclusionNum = 1;
                break;
            end
        end

%         disp(strcat('generateImageExtracts_getSortedRoisForNodule: radiologist = ', servicingRadiologistID, '; rsId = ', num2str(rsId)) );
        
        % Create a new cell matrix with the facts we need to sort...
        a = [ roiId, roiIdx, inclusionNum, unId, urnIdx, rsId_urn, rsIdx, rsId ];
        for j=1 : length(a)
            roiCellMat{i,j} = a(j);
        end
    end

    % Convert the cell matrix to a numeric matrix.
    roiCellMat = cell2mat(roiCellMat);
    % Sort the numeric matrix, first on inclusionNum, then on rsId_urn.
    sortedRoiCellMat = sortrows(roiCellMat, [3 6]);

end

