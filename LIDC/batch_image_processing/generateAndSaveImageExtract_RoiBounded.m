% generateAndSaveImageExtract_RoiBounded.m
% 
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Generates a full-size image displaying the ROIs in different colors, as supplied by radiologists, for the given RGB image,
function fRoiBounded = generateAndSaveImageExtract_RoiBounded(rgbImg, outputFolder, outputFileName, sopUid, csvData, roisForImgSopUID, dcmData, studyId, seriesId)

    % Copy the image.
    img = rgbImg;

    % PK, part 1: Extract the XMLStudyNode so we know where to look in the edgeMap file.
    xmlStudyNode = regexprep(outputFolder, '.*(LIDC-IDRI-\d\d\d\d).*', '$1');
    % PK, part 2: Extract the FileNode, so know where to look in the edgeMap file.
    fileNodeCandidateIdxs = find(strcmp(csvData.edges{4}, xmlStudyNode));
    fileNode = max(csvData.edges{7}(fileNodeCandidateIdxs));


    %%% Output image without the convex hull.
    % Create a non-displayed image.
    fRoiBounded = figure('visible','off');
    
    % For each radiologist's ROI for this particular image...
    for i = 1:numel(roisForImgSopUID)
        % Determine the ROI ID.
        roiId = roisForImgSopUID(i);
        
        % Now that we know the ROI to be applied, let's determine:
        % 1) the edges and 2) the radiologist...
        roiVerticesIdxs = find(csvData.edges{3} == roiId & strcmp(csvData.edges{4}, xmlStudyNode) & csvData.edges{7} == fileNode);
        x = csvData.edges{1}(roiVerticesIdxs);
        y = csvData.edges{2}(roiVerticesIdxs);

        
        % Determine the radiologist ID.
        roiIdx = find(csvData.roi{4} == roiId & strcmp(csvData.roi{6}, xmlStudyNode) & csvData.roi{9} == fileNode);
        inclusion = csvData.roi{3}(roiIdx);
        unId = csvData.roi{5}(roiIdx);
        urnIdx = find(csvData.unblindedReadNodule{2} == unId & strcmp(csvData.unblindedReadNodule{4}, xmlStudyNode) & csvData.unblindedReadNodule{7} == fileNode);
        rsId_urn = csvData.unblindedReadNodule{3}(urnIdx);
        rsIdx = find(csvData.readingSession{3} == rsId_urn & strcmp(csvData.readingSession{5}, xmlStudyNode) & csvData.readingSession{8} == fileNode);
        rsId = csvData.readingSession{3}(rsIdx);
        servicingRadiologistID = csvData.readingSession{2}(rsIdx);
        
        if isnumeric(servicingRadiologistID); servicingRadiologistID = num2str(servicingRadiologistID); end
        disp(strcat('roiBounded: radiologist = ', servicingRadiologistID, '; rsId = ', num2str(rsId)) );
        
        img = assignRadiologistColors(img, y, x, rsId, inclusion, false);

    end
    
    
    %% Save the image.
    % As TIFF.
    tiffFilePath = strcat( outputFolder, filesep, strrep(outputFileName, '.dcm', '-'), 'allReviewers-noConvexHull.tiff' );
    imwrite(img, tiffFilePath);
    
    % 20150317: estory: We also want DICOM output, with info copied.
    % Reading the written file instead of the in-memory data isn't efficient, but hours of googling revealed no way to write the in-memory RGB image to DICOMC format... bizarre.
    dcmFilePath = strcat( outputFolder, filesep, strrep(outputFileName, '.dcm', '-'), 'allReviewers-contours.dcm' );
    dicomwrite(imread(tiffFilePath), dcmFilePath, dcmData.info);
    
    
    %% Create or append to a CSV file containing: imageSOP_UID, file path, DICOM info.
    csvImageMappingFileName = strcat('imageSOP_UID-filePath-dicominfo-', xmlStudyNode, '.csv');
    
    % Get col names.
    nonDicomCols = [ { 'inserted_datetime', 'StudyInstanceUID', 'SeriesInstanceUID', 'XmlStudyNode', 'FileNode', 'imageSOP_UID', 'DICOM_original_fullPath', 'DICOM_contours_fullPath', 'TIFF_contours_fullPath'} ];
    f = nonDicomCols;
    flds = fieldnames(dcmData.info);
    for i=1 : numel(flds)
       f = [ f, flds{i} ];
    end
        
    % Write CSV headers.
    if ~exist(csvImageMappingFileName, 'file')
        disp(strcat('Creating DICOM CSV file: ', csvImageMappingFileName));
        fid = fopen(csvImageMappingFileName, 'w');
        fprintf(fid, '"%s",', f{1:end}) ;
        fprintf(fid, '"%s"\n', '') ;
        fclose(fid) ;
    end

    % Get col values.
    c = [];
    nonDicomValues = [ {datestr8601}, {studyId}, {seriesId}, {xmlStudyNode}, {num2str(fileNode)}, {cell2mat(sopUid)}, {dcmData.fullPath}, {dcmFilePath}, {tiffFilePath} ];
    for i=1 : numel(nonDicomValues)
        c{i} = nonDicomValues{i}; 
    end
    for i = (numel(nonDicomValues)+1):numel(f)
        c{i} = dcmData.info.(f{i});
    end
    

    for i=1 : numel(c(1,:))
       if(isstruct(c(1,i)))
           disp(strcat('Struct at: ', num2str(i), '; ', f(i)));
       end
    end
    
    % Write CSV rows.
    fid = fopen(csvImageMappingFileName, 'a') ;
    % 20150827: BREAKS ON COL 43 BECAUSE THE VALUE THERE IS A STRUCT CONTAINING:     
    %     FamilyName: ''
    %     GivenName: ''
    %     MiddleName: ''
    %     NamePrefix: ''
    %     NameSuffix: ''
    for i=1 : numel(c(1,:))
        disp(strcat('i=',num2str(i)));
        v = c(1,i);

        % 20150827: estory: The col #s at which a struct was found by trial-and-error (literally, crashing, since MATLAB's isstruct function does not detect a struct, even though disp'ing the value indicates it's a 1x1 struct).
%         if intersect([i], [43, 45, 46, 49, 109, 110, 111, 112, 113, 120])
            strFromEdgeCase = '';
            je = 1; % numel(v{1,:})
            ke = numel(v(:,1));
            for j=1 : je
                for k=1 : ke
                    disp(strcat('j=',num2str(j),'/',num2str(je),'; k=',num2str(k),'/',num2str(ke)));
                    v
disp(strcat('isstruct=', num2str(isstruct(v{k,j})), '; ischar=', num2str(ischar(v{k,j})), '; iscell=', num2str(iscell(v{k,j})), '; isinteger=',num2str(isinteger(v{k,j})), '; isa(v{k,j}, "double")=', num2str(isa(v{k,j}, 'double')), '; size=', num2str(size(v{k,j}))));
                    if ischar(v{k,j})
                        strFromEdgeCase = strcat(strFromEdgeCase, v{k,j});
%                     elseif isa(v{k,j}, 'double') || isinteger(v{k,j})
                    elseif isnumeric(v{k,j})
                        strFromEdgeCase = strcat(strFromEdgeCase, num2str(v{k,j}));
                    else


                        a = v{k,j};
                        % if this is a 2nd-level struct... then unpacking it will be a nightmare because MATLAB makes accessing struct key-value pairs in a dynamic way a fiery, unworkable hell.
                        if isstruct(a)
                            flds = fieldnames(a);
                            if numel(flds) > 0
                                if strcmpi(flds(1),'Item_1') == 1
                                    strFromEdgeCase = strcat(strFromEdgeCase, 'Item_1: TODO / IGNORED 2nd-level struct');
                                else
                                    strFromEdgeCase = strcat(strFromEdgeCase, 'TODO / IGNORED 2nd-level struct');
                                end
                            else
                                strFromEdgeCase = strcat(strFromEdgeCase, 'TODO / IGNORED 2nd-level struct (0 fields)');
                            end
                        else
                            strFromEdgeCase = strcat(strFromEdgeCase, mat2str(cell2mat(struct2cell(v{k,j}))));
                        end
                    end
                    
                end
            end
            fprintf(fid, '"%s",', strFromEdgeCase) ;
%         else
%             fprintf(fid, '"%s",', c{1,i}) ;
%         end
    end
%     fprintf(fid, '"%s",', c{1,1:end-73}) ;        
% fprintf(fid, '"%s"\n', c{1,end}) ;
    fprintf(fid, '\n');
    fclose(fid) ;

    
    %% Deallocate memory.
    clear nonDicomCols;
    clear nonDicomValues;
    clear f;
    clear c;
end

