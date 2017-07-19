% function [noduleRecall, recallRate, thresholdedMask, centerInds] = ComputeScoreMapAUC(scoreMapFolder, uniqueMaskPath, threshScoreMask, windowDim,removeSmallNodFlag)
%very similar to ComputeThresholdRecallRate, but computes AUC according to
%truthing method;
pkg load image; %this for bwlabeln
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%modelName = {'cnn_36368_20170123173328', 'cnn_36368_20170122173913','cnn_36368_20170124123325'}; %scoreMap averaged across these; 
modelName = {'cnn_36368_20170123173026', 'cnn_36368_20170124095720', 'cnn_36368_20170124090728'};%scoreMap averaged across these; 
uniqueMaskPath = '/raida/apezeshk/lung_dicom_dir/';
masterSaveFolder = fullfile('/raida/apezeshk/temp/score_map_post_proc', 'average'); %results will be written in subfolders based on truthMode and runMode
threshScoreMask = 0.1; %after non-max suppression, this threshold is applied to obtain a binary "detection" map
windowDim = [5, 5, 3]; %window dimensions when applying non-max suppression
removeSmallNodFlag = 1; %if 1, will skip small <3mm nodules in uniqueStats when determining if a coordinate is matching an actual nodule
truthMode = 'nodule'; %'nodule' or 'patch'; if nodule, peak coordinate should fall within nodule bbox for detection; if patch, it should fall within patch centered at nodule centroid (depends on training patch size)
%NOTE: for FP extraction truthMode should be set to 'patch' to make sure no "shifted" positives leak into the FPs being extracted;
% For AUC calculation, makes more sense to use truthMode='nodule', otherwise big patch sizes will always result in high AUC!
patchSize = [36, 36, 8]; %only used if truthMode=='patch'; 
runMode = 'test'; %options are 'train' or 'test'; determines subfolder of score maps to read from, and subfolder for where results will be written
%scoreMapFolder =  fullfile('/diskStation/LIDC/36368/score_map/',modelName, runMode); %'/raida/apezeshk/temp/DeepMed/scoreMaps'  %Folder to read score maps from
masterScoreMapFolder =  '/diskStation/LIDC/36368/score_map/'; %'/raida/apezeshk/temp/DeepMed/scoreMaps'  %Folder to read score maps from
lungInteriorFolder = '/raida/apezeshk/LIDC_v2/037_part_to_lung' %'/raida/apezeshk/LIDC_v2/015_lung_only'
aucFlag = 1; %if 1, also computes AUC across all cases
resizeFlag = 0; %if 1 operates based on assumption that scoreMap has been resized to full size volume (so will also use data from uniqueStats); if 0, the FCN processed nodule mask will be used as truthing source
lungInteriorFlag = 1; %if 1, multiplies scoreMask by lungInterior mask s.t. only centerpoints within the lung contribute to AUC, etc calc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files = dir(fullfile(masterScoreMapFolder, modelName{1}, runMode, '*.mat')); %use any of the models to find out relevant cases
%files = dir(fullfile('/diskStation/LIDC/36368/score_map/cnn_36368_20161209111319/train', '*.mat'));
%files(1:2) = [];
noduleCounter = 0; %keeps track of current nodule row in master cell array for detection results across all cases
noduleRecall = {};
scoresAll = [];
labelsAll = [];
cellHeader = {'CaseName', 'Tag', 'DetectionStatus', 'ModelName', 'avgPrincipalAxisLength', 'avgVolume',...
 'IURatio', 'minx', 'maxx', 'miny', 'maxy', 'minz', 'maxz'};

if ~exist(masterSaveFolder, 'dir')
        mkdir(masterSaveFolder);
end

smallNodScoreAccumalator = {};
for jj = 1:length(files)
        %if jj==22 || jj==28|| jj==24 || jj==29 || jj==42 %these are indices of high-slice cases
        %    continue
        %end

        fprintf('Loading No.%d testing subject (total %d).\n', jj,length(files));
        currentMapName = files(jj).name;
        if lungInteriorFlag == 1 %check like this, bc even if mask doesn't exist the scoreMap mat file will have an empty array in it
                  currentLungInteriorFile = currentMapName(1:end-4); %first get rid of the '.mat'
                  currentLungInteriorFile = fullfile(lungInteriorFolder, 
                    regexprep(currentLungInteriorFile, '_', '/'), 
                    ['lungInterior_', currentLungInteriorFile, '.mat']);
                  if exist(currentLungInteriorFile, 'file') ~= 2 %skip current case if the lung interior doesn't exist
                     continue
                  end                  
        end                  
        
        tempScoreMapFolder = fullfile(masterScoreMapFolder, modelName{1}, runMode); %use this just to find size to pre-set
        tempScoreMapData = load(fullfile(tempScoreMapFolder, currentMapName));
        currentScoreMap = zeros(size(tempScoreMapData.scoreMap));
        for kk = 1:length(modelName)
          currentScoreMapFolder = fullfile(masterScoreMapFolder, modelName{kk}, runMode); 
          currentScoreMapData = load(fullfile(currentScoreMapFolder, currentMapName));          
          currentScoreMap = currentScoreMap + currentScoreMapData.scoreMap;
        end
        currentScoreMap = currentScoreMap/length(modelName); %now average the models
%         currentScoreMapData = currentScoreMapData.vect;
%        currentScoreMap = currentScoreMapData.scoreMap;
        if lungInteriorFlag == 1 %in this case, extract only lung portion of scoreMap
            currentLungInteriorMask = logical(currentScoreMapData.lungInteriorMask);
            %for kk = 1:size(currentLungInteriorMask,3)
            %   se = strel('square', 3);
            %   currentLungInteriorMask(:,:,kk) = imerode(currentLungInteriorMask(:,:,kk), se);
            %end
            currentScoreMap = currentScoreMap.*currentLungInteriorMask;
        end
            
        
        %do 3d max suppression for current case
        [maxmap, maxidx] = minmaxfilt(currentScoreMap, windowDim, 'max', 'same');
        bool = (currentScoreMap == maxmap);
        currentMapMaxSupp = single(bool).*currentScoreMap;     
        
        %Note that centerInds indexing is matlab style, i.e. starting from
        %1; layer this will be accounted for when generating fp's in python
        [thresholdedMask, centerInds] = GetCandidatesFromScoreMap(currentMapMaxSupp, threshScoreMask);   
        
        if resizeFlag == 1
                % in this mode, the scoreMap has been resized to full size,
                % and all comparisons for truthing are done against data in
                % uniqueStats, which is also full size;
                folderComp = regexprep(currentMapName, '_', '/');
                folderComp = regexprep(folderComp, '.mat', '');
                currentUniquePath = fullfile(uniqueMaskPath, folderComp, ['uniqueStats_' currentMapName]);
                uniqueMaskData = load(currentUniquePath);
                %         noduleMask = uniqueMaskData.allMaxRadiologistMsk;
                uniqueStats = uniqueMaskData.uniqueStats;

                currentNegativeInd = 1:size(centerInds,1); %initialize; list of all row numbers for peaks in centerInds; later these will be reduced by taking out the row numbers corresponding to positive peaks

                for m = 1:length(uniqueStats)
                        current_avgPrincipalAxisLength = uniqueStats(m).avgPrincipalAxisLength;
                        %                         if current_avgPrincipalAxisLength == 0:
                        %                                 current_minx = uniqueStats(m).minX -1; %extend the min/max in x,y to make sure the thing gets obliterated
                        %                                 current_maxx = uniqueStats(m).maxX + 1;
                        %                                 current_miny = uniqueStats(m).minY - 1;
                        %                                 current_maxy = uniqueStats(m).maxY + 1;
                        %                                 current_minz = uniqueStats(m).minZ; %somehow z just does not need to be fixed
                        %                                 current_maxz = uniqueStats(m).maxZ;
                        %                                 %swap x,y to get matrix coordinates bc matlab x,y are image coordinates
                        %                                 noduleMask(current_miny:current_maxy, current_minx:current_maxx, current_minz:current_maxz) = 0;
                        %                         end
                        if removeSmallNodFlag == 1 %whether to ignore the <3mm nodules from the master nodule mask
                                if current_avgPrincipalAxisLength == 0
                                        continue
                                end
                        end
                        noduleCounter = noduleCounter + 1;

                        noduleRecall{noduleCounter, 1} =  currentMapName;
                        noduleRecall{noduleCounter, 2} = m; %this is the nodule unique tag for that case
                        noduleRecall{noduleCounter, 3} = 0; %initialize; later, if this nodule is detected in thresholded max suppressed score map, will be set to 1
                        if strcmp(truthMode, 'nodule')
                                bounds_minx = uniqueStats(m).minX; %note these are matlab image coordinates, so x,y are flipped relative to row,column
                                bounds_maxx = uniqueStats(m).maxX;
                                bounds_miny = uniqueStats(m).minY;
                                bounds_maxy = uniqueStats(m).maxY;
                                bounds_minz = uniqueStats(m).minZ;
                                bounds_maxz = uniqueStats(m).maxZ;

                        elseif strcmp(truthMode, 'patch')
                                current_centroidx = uniqueStats(m).avgVolCentroidX;%note these are matlab image coordinates, so x,y are flipped relative to row,column
                                current_centroidy = uniqueStats(m).avgVolCentroidY;
                                current_centroidz = uniqueStats(m).avgVolCentroidZ;
                                bounds_minx = min(floor(current_centroidx - patchSize(1)/2), uniqueStats(m).minX);%note these are matlab image coordinates, so x,y are flipped relative to row,column
                                bounds_maxx = max(ceil(current_centroidx + patchSize(1)/2), uniqueStats(m).maxX);
                                bounds_miny = min(floor(current_centroidy - patchSize(2)/2), uniqueStats(m).minY);
                                bounds_maxy = max(ceil(current_centroidy + patchSize(2)/2), uniqueStats(m).maxY);
                                bounds_minz = min(floor(current_centroidz - patchSize(3)/2), uniqueStats(m).minZ);
                                bounds_maxz = max(ceil(current_centroidz + patchSize(3)/2), uniqueStats(m).maxZ);
                        end
                        currentNoduleCheck = ismember(centerInds(:,2), bounds_minx:bounds_maxx) & ismember(centerInds(:,1), bounds_miny:bounds_maxy) & ismember(centerInds(:,3), bounds_minz:bounds_maxz);
                        currentNoduleMatchInd = find(currentNoduleCheck); %row indices of peak points that fall within bounds of current nodule
                        currentNegativeInd = setdiff(currentNegativeInd, currentNoduleMatchInd); %sequentially take out row indices of peaks that match a nodule in current case
                        if ~isempty(find(currentNoduleCheck))
                                noduleRecall{noduleCounter, 3} = 1;
                        end
                        %cellHeader = {'CaseName', 'Tag', 'DetectionStatus', 'ModelName', 'avgPrincipalAxisLength', 'avgVolume',...
                        %'IURatio', 'minx', 'maxx', 'miny', 'maxy', 'minz', 'maxz'};%cellHeader repeated here to make sure ordering is correct below
                        noduleRecall{noduleCounter, 4} = modelName;
                        noduleRecall{noduleCounter, 5} = uniqueStats(m).avgPrincipalAxisLength;
                        noduleRecall{noduleCounter, 6} = uniqueStats(m).avgVolume;
                        noduleRecall{noduleCounter, 7} = uniqueStats(m).IURatio;
                        noduleRecall{noduleCounter, 8} = uniqueStats(m).minX;
                        noduleRecall{noduleCounter, 9} = uniqueStats(m).maxX;
                        noduleRecall{noduleCounter, 10} = uniqueStats(m).minY;
                        noduleRecall{noduleCounter, 11} = uniqueStats(m).maxY;
                        noduleRecall{noduleCounter, 12} = uniqueStats(m).minZ;
                        noduleRecall{noduleCounter, 13} = uniqueStats(m).maxZ;
                end
                
                

                
        elseif resizeFlag == 0
                %in this mode the raw (unresized) FCN processed scoreMap is read from
                %the mat file, and comparison for truthing is done against
                %the noduleMask that has been processed with similar 
                %architecture FCN (so it has same size as the scoreMap);
                %Assumes the option to tag the nodules prior to passing
                %them thru FCN is used, s.t. each nodules has same tag as
                %corresponding nodule in uniqueStats; If not, uncomment the
                %marked section, which works on assumption of a binary
                %noduleMask (untagged), but we won't know which nodule is
                %which in that case;
%                 if ~strcmp(truthMode, 'nodule') %truthMode=='patch' doesn't make sense when resizeFlag==0
%                         error('Incompatible options: resizeFlag==0 can only be used with truthMode==nodule!');
%                 end
                
                folderComp = regexprep(currentMapName, '_', '/');
                folderComp = regexprep(folderComp, '.mat', '');
                currentUniquePath = fullfile(uniqueMaskPath, folderComp, ['uniqueStats_' currentMapName]);
                %disp(currentUniquePath);
                uniqueMaskData = load(currentUniquePath);
                %         noduleMask = uniqueMaskData.allMaxRadiologistMsk;
                uniqueStats = uniqueMaskData.uniqueStats;
                
                noduleMask = currentScoreMapData.noduleMask; %this is FCN processed noduleMask, not the full-size one!
                noduleMask_lessThan3mm = currentScoreMapData.noduleMask_lessThan3mm; %this is FCN processed <3mm nodule mask

                [h, w, z] = size(noduleMask);

                currentNegativeInd = 1:size(centerInds,1); %initialize; list of all row numbers for peaks in centerInds; later these will be reduced by taking out the row numbers corresponding to positive peaks
                currentSmallNodNegativeInd = 1:size(centerInds,1); %initialize; list of all row numbers for peaks in centerInds; later these will be reduced by taking out the row numbers corresponding to + peaks from small nodules only (may have overlap with >3mm + peaks)
                for m = 1:length(uniqueStats)
                        current_avgPrincipalAxisLength = uniqueStats(m).avgPrincipalAxisLength;
                        if removeSmallNodFlag == 1 %whether to ignore the <3mm nodules from the master nodule mask
                                %The FCN may detect <3mm nodules; To avoid corrupting AUC calculation, first remove any center inds that fall within areas belonging to <3mm nodules
                                %Then skip the ops in rest of the loop because the <3mm nodules shouldn't count towards calculation of recall and labeling either.
                                if current_avgPrincipalAxisLength == 0
                                        currentNoduleInd = find(noduleMask_lessThan3mm == m); %Note comparison is against appropriate mask, i.e. one with <3mm nodules in it!
                                        [row, col, slc] = ind2sub([h, w, z], currentNoduleInd);
                                        bounds_minrow = min(row(:)); %note unlike other condition resizeFlag==1 which had matlab image coordinates for these, x,y,z here are actual row,column, slice (not flipped)
                                        bounds_maxrow = max(row(:));
                                        bounds_mincol = min(col(:));
                                        bounds_maxcol = max(col(:));
                                        bounds_minslc = min(slc(:));
                                        bounds_maxslc = max(slc(:));
                                        currentNoduleCheck = ismember(centerInds(:,1), bounds_minrow:bounds_maxrow) & ismember(centerInds(:,2), bounds_mincol:bounds_maxcol) & ismember(centerInds(:,3), bounds_minslc:bounds_maxslc);
                                        currentNoduleMatchInd = find(currentNoduleCheck); %row indices of peak points that fall within bounds of current nodule
                                        %Note that some <3mm are marked by another radiologist as >3mm
                                        currentSmallNodNegativeInd = setdiff(currentSmallNodNegativeInd, currentNoduleMatchInd); %sequentially take out row indices of peaks that match a <3mm nodule in current case; 
                                        
                                        continue
                                end
                        end
                        %the >3mm nodules get processed here
                        noduleCounter = noduleCounter + 1;
                                                
                        noduleRecall{noduleCounter, 1} =  currentMapName;
                        noduleRecall{noduleCounter, 2} = m; %this is the nodule tag for that case; RIGHT NOW TAG IS ACCORDING TO BWLABELN, AND NOT TAG NUMBER FROM UNIQUESTATS!!!
                        noduleRecall{noduleCounter, 3} = 0; %initialize; later, if this nodule is detected in thresholded max suppressed score map, will be set to 1
                        
                        currentNoduleInd = find(noduleMask == m);
                        [row, col, slc] = ind2sub([h, w, z], currentNoduleInd);
                        if strcmp(truthMode, 'nodule')
                                bounds_minrow = min(row(:)); %note unlike other condition resizeFlag==1 which had matlab image coordinates for these, x,y,z here are actual row,column, slice (not flipped)
                                bounds_maxrow = max(row(:));
                                bounds_mincol = min(col(:));
                                bounds_maxcol = max(col(:));
                                bounds_minslc = min(slc(:));
                                bounds_maxslc = max(slc(:));
                        elseif strcmp(truthMode, 'patch') 
                                %In patch mode, use a somewhat wider bbox
                                %to make sure as much as possible that fp patches extracted later won't contain whole or edges of nodules;
                                %The values subtracted/added are kind of
                                %arbitrary, just to counter the aforementioned effect!
                                bounds_minrow = min(row(:)) - 6; %note unlike other condition resizeFlag==1 which had matlab image coordinates for these, x,y,z here are actual row,column, slice (not flipped)
                                bounds_maxrow = max(row(:)) + 6;
                                bounds_mincol = min(col(:)) - 6;
                                bounds_maxcol = max(col(:)) + 6;
                                bounds_minslc = min(slc(:)) - 3;
                                bounds_maxslc = max(slc(:)) + 3;
                        end
                        
                        currentNoduleCheck = ismember(centerInds(:,1), bounds_minrow:bounds_maxrow) & ismember(centerInds(:,2), bounds_mincol:bounds_maxcol) & ismember(centerInds(:,3), bounds_minslc:bounds_maxslc);
                        currentNoduleMatchInd = find(currentNoduleCheck); %row indices of peak points that fall within bounds of current nodule
                        currentNegativeInd = setdiff(currentNegativeInd, currentNoduleMatchInd); %sequentially take out row indices of peaks that match a nodule in current case
                        if ~isempty(find(currentNoduleCheck))
                                noduleRecall{noduleCounter, 3} = 1;
                        end
                        noduleRecall{noduleCounter, 4} = modelName;
                        noduleRecall{noduleCounter, 5} = uniqueStats(m).avgPrincipalAxisLength;
                        noduleRecall{noduleCounter, 6} = uniqueStats(m).avgVolume;
                        noduleRecall{noduleCounter, 7} = uniqueStats(m).IURatio;
                        noduleRecall{noduleCounter, 8} = uniqueStats(m).minX;
                        noduleRecall{noduleCounter, 9} = uniqueStats(m).maxX;
                        noduleRecall{noduleCounter, 10} = uniqueStats(m).minY;
                        noduleRecall{noduleCounter, 11} = uniqueStats(m).maxY;
                        noduleRecall{noduleCounter, 12} = uniqueStats(m).minZ;
                        noduleRecall{noduleCounter, 13} = uniqueStats(m).maxZ;
                end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%This is old way for how the different ppties were
%%%%%extracted and labeling was done; It was before tagging
%%%%%of nodules in noduleMask was implemented, so nodule tags couldn't be identified 
%%%%%and therefore everything was done based on the binary noduleMask
%%%%%processed thru the FCN. Also, recall is calculated off of nodules that
%%%%%passed thru the FCN with something from them remaining, so if the FCN
%%%%%makes a nodule disappear that nodule won't be in total count of
%%%%%nodules to calculate recall for! But why would max pooling make a
%%%%%nodule disappear altogether?! maybe because adjacent nodules across different dimensions will be
%%%%%merged together.
%                 [noduleMaskLabeled, numObjects] = bwlabeln(logical(noduleMask)); %has to be something other than int64
%                 currentNegativeInd = 1:size(centerInds,1); %initialize; list of all row numbers for peaks in centerInds; later these will be reduced by taking out the row numbers corresponding to positive peaks
%                 for m = 1:numObjects
%                         noduleCounter = noduleCounter + 1;
%                         noduleRecall{noduleCounter, 1} =  currentMapName;
%                         noduleRecall{noduleCounter, 2} = m; %this is the nodule tag for that case; RIGHT NOW TAG IS ACCORDING TO BWLABELN, AND NOT TAG NUMBER FROM UNIQUESTATS!!!
%                         noduleRecall{noduleCounter, 3} = 0; %initialize; later, if this nodule is detected in thresholded max suppressed score map, will be set to 1
%                         [h, w, z] = size(noduleMask);
%                         currentNoduleInd = find(noduleMaskLabeled == m);
%                         [row, col, slc] = ind2sub([h, w, z], currentNoduleInd);
%                         bounds_minrow = min(row(:)); %note unlike other condition resizeFlag==1 which had matlab image coordinates for these, x,y,z here are actual row,column, slice (not flipped)
%                         bounds_maxrow = max(row(:));
%                         bounds_mincol = min(col(:));
%                         bounds_maxcol = max(col(:));
%                         bounds_minslc = min(slc(:));
%                         bounds_maxslc = max(slc(:));
%                         currentNoduleCheck = ismember(centerInds(:,1), bounds_minrow:bounds_maxrow) & ismember(centerInds(:,2), bounds_mincol:bounds_maxcol) & ismember(centerInds(:,3), bounds_minslc:bounds_maxslc);
%                         currentNoduleMatchInd = find(currentNoduleCheck); %row indices of peak points that fall within bounds of current nodule
%                         currentNegativeInd = setdiff(currentNegativeInd, currentNoduleMatchInd); %sequentially take out row indices of peaks that match a nodule in current case
%                         if ~isempty(find(currentNoduleCheck))
%                                 noduleRecall{noduleCounter, 3} = 1;
%                         end
%                         noduleRecall{noduleCounter, 4} = modelName;
%                         noduleRecall{noduleCounter, 5} = 0; %was avgPrincipalAxis in resizeFlag==1
%                         noduleRecall{noduleCounter, 6} = 0;%was avgVolume in resizeFlag==1
%                         noduleRecall{noduleCounter, 7} = 0;%was IURatio in resizeFlag==1
%                         %Note unlike other condition, these are coordinates in FCN proc'ed noduleMask (i.e. instead of being based on uniqeStats)
%                         noduleRecall{noduleCounter, 8} = bounds_mincol; %switching to matlab image coordinates for boundaries!
%                         noduleRecall{noduleCounter, 9} = bounds_maxcol;
%                         noduleRecall{noduleCounter, 10} = bounds_minrow;
%                         noduleRecall{noduleCounter, 11} = bounds_maxrow;
%                         noduleRecall{noduleCounter, 12} = bounds_minslc;
%                         noduleRecall{noduleCounter, 13} = bounds_maxslc;
%                 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
        
        if removeSmallNodFlag == 0
                currentCaseLabels = ones(1, size(centerInds,1)); %initialize row vector of labels for current case
                currentCaseLabels(currentNegativeInd) = 0;
                currentCaseScores = zeros(1, size(centerInds, 1));
                for ii = 1:length(currentCaseScores)
                        currentCaseScores(ii) = currentMapMaxSupp(centerInds(ii,1), centerInds(ii,2), centerInds(ii,3));
                end
                
        elseif removeSmallNodFlag== 1
                %Indices corresponding to detected <3mm nodules should be
                %removed. We don't tag them as label 1, so they may have
                %high AUCs with label 0 which would skew AUC calculation.
                currentCaseLabels = ones(1, size(centerInds,1)); %initialize row vector of labels for current case
                currentCaseLabels(currentNegativeInd) = 0;
                currentCaseScores = zeros(1, size(centerInds, 1));
                for ii = 1:length(currentCaseScores)
                        currentCaseScores(ii) = currentMapMaxSupp(centerInds(ii,1), centerInds(ii,2), centerInds(ii,3));
                end
                                
                %the term in the setdiff is list of all detected <3mm nodules; but some of them may have been tagged as >3mm
                %nodules also by different radiologist, so we want to remove indices (we want to remove them completely from any
                %AUC calculation) corresponding to only those <3mm nodules that don't have a corresponding >3mm match (with a
                %different tag id)
                smallNodInds = intersect(currentNegativeInd, setdiff(1:size(centerInds,1),currentSmallNodNegativeInd));
                smallNodScoreAccumalator{jj} = currentCaseScores(smallNodInds);
                
                currentCaseLabels(smallNodInds) = []; %now exclude from further processing any center points from <3mm nodules
                currentCaseScores(smallNodInds) = [];
                centerInds(smallNodInds, :) = [];
                                
        end
        currentCaseScores = reshape(currentCaseScores, size(currentCaseLabels)); 
        
         if aucFlag==1
                  scoresAll = [scoresAll, currentCaseScores];
                  labelsAll = [labelsAll, currentCaseLabels];
         end


        disp(['Number of center peaks in current map: ' num2str(size(centerInds, 1))]);
        fflush(stdout);
        
        if ~exist(fullfile(masterSaveFolder, runMode), 'dir')
            mkdir(fullfile(masterSaveFolder, runMode));
        end
        if ~exist(fullfile(masterSaveFolder, runMode, truthMode), 'dir')
            mkdir(fullfile(masterSaveFolder, runMode, truthMode));
        end
            
        save('-v7', fullfile(masterSaveFolder, runMode, truthMode, currentMapName), 'centerInds', 'currentCaseScores', ...
        'currentCaseLabels', 'patchSize','threshScoreMask','windowDim', 'thresholdedMask','truthMode','resizeFlag', 'removeSmallNodFlag', 'runMode');
end

%now compute AUC across scores/labels from all cases
mask_class0 = labelsAll==0;
mask_class1 = labelsAll==1;
if aucFlag==1
      [AUC,S] = fastDeLong_mod(scoresAll(mask_class0),scoresAll(mask_class1));
end

recallRate = length(find([noduleRecall{:,3}]))/size(noduleRecall,1);

save('-v7', fullfile(masterSaveFolder, runMode, truthMode,'PostProcAUC.mat'), 'masterScoreMapFolder','patchSize',...
        'threshScoreMask','windowDim', 'truthMode','removeSmallNodFlag','labelsAll', 'scoresAll',...
        'resizeFlag', 'AUC','recallRate','noduleRecall', 'smallNodScoreAccumalator', 'runMode', ...
        'lungInteriorFlag', 'modelName');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if you want to get rid of large constant areas that occur in the corners
% of many CT slices, you can use the code below to eliminate the large
% areas in each slice according to a size threshold;
% zxc = currentMapMaxSupp > 0; %make a binary version that can be input to bwlabel and regionprops
% area_thresh = 30; %threshold for area of objects in each slice; anything larger will get eliminated
% for i = 1:size(currentMapMaxSupp, 3)        
%         qwe = bwlabel(zxc(:,:,i)); 
%         s_qwe = regionprops(qwe, 'ConvexArea'); 
%         areas_qwe =[s_qwe(:).ConvexArea]; %vector of areas of objects in current slice
%         labelsToRemove = find(areas_qwe>area_thresh);
%         if length(labelsToRemove)>0
%                 for j = 1:length(labelsToRemove)
%                         qwe(qwe==labelsToRemove(j)) = 0;
%                         zxc(:,:,i) = qwe;
%                 end
%         end
% end
% currentMapMaxSupp(~zxc) = 0; %set objects that are larger than threshold to zero
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






