% genPatchSet.m
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%% Generates the patch set for a DICOM image (slice), along with a statistics file for the patient in which stats about each patch are given.
function nxn = genPatchSet(folderRoot, fileName, roiId, rsId, origMat, binMat)
    n = 64;
    nxn = n*n;

    % Create folder paths.
    patchRoot = fullfile(strcat(folderRoot, filesep), 'patches');
    patchRootOrig = fullfile(strcat(folderRoot, filesep), 'patches', filesep, 'orig');
    mkdir(patchRootOrig);
    patchRootBin = fullfile(strcat(folderRoot, filesep), 'patches', filesep, 'bin');
    mkdir(patchRootBin);
    
    % Setup dimensions.
    height = size(origMat, 1);
    width = size(origMat, 2);
    wUnit = (width / n);
    hUnit = (height / n);
    
    % Get the base file name, excluding the folder path and file extension.
    [a, pfnPrefix, b] = fileparts(fileName);
    
    % Open statistics file on a per-patient basis for writing. Format: CSV, with strings double-quoted.
    fid = fopen(fullfile(strcat(patchRoot, filesep, pfnPrefix, '-bin_patch_stats', '.csv')), 'w');
    c = {'filename', 'mean', 'std'};
    fprintf(fid, '"%s",', c{1});
    fprintf(fid, '"%s",', c{2});
    fprintf(fid, '"%s"\r\n', c{3});
    
    % Loop over x vals.
    for i=0 : (n - 1)
        % Loop over y vals.
        for j=0 : (n - 1)
            % Compute indices in the image.
            pxs = wUnit * (j)+1;
            pys = hUnit * (i)+1;
            pxe = wUnit * (j + 1);
            pye = hUnit * (i + 1);

            %   Create the patch file name.
            pfnPostfix = strcat('-', int2str(pxs), ',', int2str(pys), '_to_', int2str(pxe), ',', int2str(pye));  % disp(int2str(pye));
            outFileName = strcat(pfnPrefix, pfnPostfix, '-roi_', num2str(roiId), '-rs_', num2str(rsId), '.tiff');

            %% Original image
            % Get & write the patch for the original image.
            p = origMat(pxs:pxe, pys:pye);
            p = mat2gray(p);
            outFilePathOrig = fullfile(strcat(patchRootOrig, filesep, outFileName));
            imwrite(p, outFilePathOrig);
            
            %% Binary image
            % Get & write the patch for the binary image.
            p = binMat(pxs:pxe, pys:pye);
            outFilePathBin = fullfile(strcat(patchRootBin, filesep, outFileName));
            imwrite(p, outFilePathBin);
            
            % Compute stats about the binary patch.
            binMean = mean(mean(p));
            binStd = std(std(p));
            
            % Write mean & std dev of the binary image to the stats file.
            line = {outFileName, binMean, binStd};
            fprintf(fid, '"%s",', line{1});
            fprintf(fid, '%3.10f,', (line{2}));
            fprintf(fid, '%3.10f\r\n', (line{3}));
        end
    end
    
    % Close statistics file.
    fclose(fid);
    
end