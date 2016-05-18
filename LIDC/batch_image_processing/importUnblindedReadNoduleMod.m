function [noduleID1,unblindedReadNodule_Id1,readingSession_Id1,XmlStudyNode1,StudyInstanceUID1,SeriesInstanceUID1,FileNode1] = importUnblindedReadNoduleMod(filename, startRow, endRow)
%IMPORTFILE Import numeric data from a text file as column vectors.
%   [NODULEID1,UNBLINDEDREADNODULE_ID1,READINGSESSION_ID1,XMLSTUDYNODE1,STUDYINSTANCEUID1,SERIESINSTANCEUID1,FILENODE1]
%   = IMPORTFILE(FILENAME) Reads data from text file FILENAME for the
%   default selection.
%
%   [NODULEID1,UNBLINDEDREADNODULE_ID1,READINGSESSION_ID1,XMLSTUDYNODE1,STUDYINSTANCEUID1,SERIESINSTANCEUID1,FILENODE1]
%   = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [noduleID1,unblindedReadNodule_Id1,readingSession_Id1,XmlStudyNode1,StudyInstanceUID1,SeriesInstanceUID1,FileNode1]
%   = importfile('unblindedReadNodule_mod.csv',2, 21376);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2015/02/18 01:30:01

%% Initialize variables.
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% Read columns of data as strings:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%s%s%s%s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric strings to numbers.
% Replace non-numeric strings with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = dataArray{col};
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[2,3,7]
    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch me
        end
    end
end

%% Split data into numeric and cell columns.
rawNumericColumns = raw(:, [2,3,7]);
rawCellColumns = raw(:, [1,4,5,6]);


%% Exclude rows with non-numeric cells
J = ~all(cellfun(@(x) (isnumeric(x) || islogical(x)) && ~isnan(x),rawNumericColumns),2); % Find rows with non-numeric cells
rawNumericColumns(J,:) = [];
rawCellColumns(J,:) = [];

%% Allocate imported array to column variable names
noduleID1 = rawCellColumns(:, 1);
unblindedReadNodule_Id1 = cell2mat(rawNumericColumns(:, 1));
readingSession_Id1 = cell2mat(rawNumericColumns(:, 2));
XmlStudyNode1 = rawCellColumns(:, 2);
StudyInstanceUID1 = rawCellColumns(:, 3);
SeriesInstanceUID1 = rawCellColumns(:, 4);
FileNode1 = cell2mat(rawNumericColumns(:, 3));

