% batchImageProc.m (or batchImageProcES.m)
%
% Original author: Evan Story (estory1@gmail.com)
%
% Purpose:
%
%   Original (ca. Dec. 2014): Superimposes the ROI of an image onto that image given the edges, and saves the resulting image file.
%
%   Current: (20140217): Generate crops (BW, BG, FG) of the nodules as separate files, for each radiologist's reading session, as well as a full-size image displaying all ROIs.

% LIDC-IDRI-nnnn subset to handle... this is totally crappy - but it's a workaround for:
%   1) Not having a compute cluster across which to divide-and-conquer w/ a distributed algo.
%   2) MATLAB eventually running out of memory due to some still mysterious memory leak (despite my manual freeing),
%   and this script needing to be restarted from, and including, the failure point.

% for values < 800, I'm giving ranges estimated based on output rate for each workstation after 30h, with the expectation that I will be able to adjust the workstation sets after about another 30 hours.
regexSubsetToHandle = 'LIDC-IDRI-(0[89]\d\d|[1-9]\d\d\d|07[2-9]\d|06[1-9]\d|05[3-9]\d|04[2-9]\d|03[3-9]\d|02[5-9]\d)';

% Get the executing machine's hostname.
[~, computerName] = system('hostname');

% If is my laptop...
if ismac
    disp(strcat('* Running on Mac, for subset in regex: ', regexSubsetToHandle));
    inputFolderRootBase = fullfile(pwd, '..', '..', 'LIDC_Complete_20141106', 'LIDC-IDRI');
    csvFolderFullPath = fullfile(pwd, '..', '..', 'LIDC_Complete_20141106', 'Export_CT');
    baseOutputFolderRoot = pwd;
% else, if medixsrv
elseif strcmpi('MEDIXSRV', strtrim(computerName)) == 1
    disp(strcat('* Running on Medixsrv, for subset in regex: ', regexSubsetToHandle));
    inputFolderRootBase = fullfile('D:\LIDC\LIDC_Complete_20141106\LIDC-IDRI');
    csvFolderFullPath = fullfile('D:\LIDC\LIDC_Complete_20141106\Export_CT');
    baseOutputFolderRoot = pwd;
% else, assume Medix lab workstation
else
    disp(strcat('* Running on non-Mac, non-Medixsrv host (e.g. Medix workstation; hostname is "', computerName, '") for subset in regex: ', regexSubsetToHandle));
    inputFolderRootBase = fullfile('\\medixsrv\LIDC\LIDC_Complete_20141106\LIDC-IDRI');
    csvFolderFullPath = fullfile('\\medixsrv\LIDC\LIDC_Complete_20141106\Export_CT');
    baseOutputFolderRoot = fullfile('\\medixsrv\Workspace\Evan\StoryEvan\edge_superimposition');
end

% Recursively get all image folders. src: http://stackoverflow.com/questions/20284377/matlab-list-all-unique-subfolders
[success,message,messageid] = fileattrib(strcat(inputFolderRootBase, filesep, '*'));
isfolder = [message(:).directory];
[folders{1:sum(isfolder)}] = deal(message(isfolder).Name);
isDeepest = cellfun(@(str) numel(strmatch(str,folders))==1, folders);
deepestFolders = folders(isDeepest);


%% Define the CSV file names in the input folders.
readingSessionFileFullPath = fullfile(csvFolderFullPath, 'readingSession.csv');
unblindedReadFileFullPath = fullfile(csvFolderFullPath, 'unblindedReadNodule_mod.csv');
roiFileFullPath = fullfile(csvFolderFullPath, 'roi.csv');
verticesFileFullPath = fullfile(csvFolderFullPath, 'edgeMap.csv');


%%% Read CSV files...
if (exist('readingSession', 'var') ~= 1) || (exist('unblindedReadNodule','var') ~= 1) || (exist('roi', 'var') ~= 1) || (exist('edges', 'var') ~= 1)
    
    %% Read the reading session file.
    disp(strcat('Reading CSV (reading session): ',readingSessionFileFullPath));
    [annotationVersion,servicingRadiologistID,RS_readingSession_Id,LidcReadMessage_Id,RS_XmlStudyNode,RS_StudyInstanceUID,RS_SeriesInstanceUID,RS_FileNode] = importReadingSession(readingSessionFileFullPath);
    readingSession = [{annotationVersion},{servicingRadiologistID},{RS_readingSession_Id},{LidcReadMessage_Id}, {RS_XmlStudyNode}, {RS_StudyInstanceUID}, {RS_SeriesInstanceUID}, {RS_FileNode}];

    %% Read the unblinded read file.
    disp(strcat('Reading CSV (unblinded read nodule): ',unblindedReadFileFullPath));
    [noduleID,unblindedReadNodule_Id,URN_readingSession_Id,URN_XmlStudyNode,URN_XmlStudyNodeStudyInstanceUID,URN_XmlStudyNodeSeriesInstanceUID,URN_XmlStudyNodeFileNode] = importUnblindedReadNoduleMod(unblindedReadFileFullPath);
    unblindedReadNodule = [{noduleID},{unblindedReadNodule_Id},{URN_readingSession_Id}, {URN_XmlStudyNode}, {URN_XmlStudyNodeStudyInstanceUID}, {URN_XmlStudyNodeSeriesInstanceUID}, {URN_XmlStudyNodeFileNode}];

    %% Read the ROI file.
    disp(strcat('Reading CSV (ROI): ',roiFileFullPath));
    [imageZposition,imageSOP_UID,inclusion,ROI_roi_Id,ROI_unblindedReadNodule_Id,ROI_XmlStudyNode,ROI_StudyInstanceUID,ROI_SeriesInstanceUID,ROI_FileNode] = importRoi(roiFileFullPath);
    % remove the double quotes, if they exist.
    for i=1 : size(imageSOP_UID, 1)
        imageSOP_UID(i) = strrep(imageSOP_UID(i), '"', '');
    end
    for i=1 : size(inclusion, 1)
        inclusion(i) = strrep(inclusion(i), '"', '');
    end
    roi = [{imageZposition},{imageSOP_UID},{inclusion},{ROI_roi_Id},{ROI_unblindedReadNodule_Id},{ROI_XmlStudyNode}, {ROI_StudyInstanceUID},{ROI_SeriesInstanceUID}, {ROI_FileNode}];


    %% Read the vertices (ROI "edges") file.
    disp(strcat('Reading CSV (vertices): ',verticesFileFullPath));
    [xCoord,yCoord,EM_roi_Id,EM_XmlStudyNode,EM_StudyInstanceUID,EM_SeriesInstanceUID,EM_FileNode] = importEdgeMap(verticesFileFullPath);
    edges = [{xCoord}, {yCoord}, {EM_roi_Id}, {EM_XmlStudyNode}, {EM_StudyInstanceUID}, {EM_SeriesInstanceUID}, {EM_FileNode}];

    % create structure to make passing-around the CSV data sane.
    csvData = struct('readingSession', {readingSession}, 'unblindedReadNodule', {unblindedReadNodule}, 'roi', {roi}, 'edges', {edges});

end


%% For each image folder...
for imageInputFolderRootFullPath_cell = deepestFolders
    % Convert from cell to char.
    imageInputFolderRootFullPath = fullfile(cell2mat(imageInputFolderRootFullPath_cell));

    % MATLAB has a memory leak, despite manual freeing; this causes MATLAB to eventually die after processing some subset of the LIDC images.
    % Here, I use a regex pattern to conditionally process the incomplete subset.
    if length(regexp(imageInputFolderRootFullPath, regexSubsetToHandle)) == 1
        % Extract the variable part of the input path, then append it to the present working folder to use as an output destination.
        outputFolderSubpath = strrep(imageInputFolderRootFullPath, inputFolderRootBase, '');
        % Compute the full output folder path.
        outputFolderFullPath = fullfile(baseOutputFolderRoot, 'out', outputFolderSubpath);
        % Make the output folder, using the varying portion of the input folder path as the new path (enables 1:1 mapping between output folder and input folder).
        mkdir(outputFolderFullPath);
        % If a previous execution generated any images, delete them. (Keep the output folder tidy.)
%         delete(fullfile(outputFolderFullPath, '*.tiff'));


        % Generate the extracts.
        disp(strcat('Generating crops for:', imageInputFolderRootFullPath));
        generateImageExtractsForStudy(imageInputFolderRootFullPath, outputFolderFullPath, csvData);         %, verticesFileFullPath, roiFileFullPath, unblindedReadFileFullPath, readingSessionFileFullPath);

    end
end