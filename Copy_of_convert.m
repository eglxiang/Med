%% Copyright (C) 2017 Xiang Xiang
%% 
%% This program is free software; you can redistribute it and/or modify it
%% under the terms of the GNU General Public License as published by
%% the Free Software Foundation; either version 3 of the License, or
%% (at your option) any later version.
%% 
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% -*- texinfo -*- 
%% @deftypefn {Function File} {@var{retval} =} convert (@var{input1}, @var{input2})
%%
%% @seealso{}
%% @end deftypefn

%% Author: Xiang Xiang <eglxiang@gmail.com>
%% Created: 2017-07-12
clear all;
close all;
alpha = 0.80;
beta = 0.70;
rootpath = '/raida/apezeshk/lung_dicom_dir/';
savepath = '/media/shamidian/sda2/ws1/mat_resized2/';
cd(rootpath);
exterior_folders=dir;
[num_exterior_folders,~]=size(exterior_folders);
for i=83:num_exterior_folders %5 %84 %133 %165 %311 %326 %327 %369 %446 %511 %545 %569
  current_exterior_folder=exterior_folders(i).name;
  cd(current_exterior_folder);
  sub_folders=dir;
  [num_sub_folders,~]=size(sub_folders);
  for j=3:num_sub_folders
    current_sub_folder=sub_folders(j).name;
    cd(current_sub_folder);
    interior_folders=dir;
    [num_interior_folder,~]=size(interior_folders);
    for k=3:num_interior_folder
      current_interior_folder=interior_folders(k).name;
      cd(current_interior_folder);
      matname = strcat('uniqueStats_',current_exterior_folder,'_',current_sub_folder,'_',current_interior_folder,'.mat');
      load(matname);
      [height,width,number]=size(allMaxRadiologistMsk);     
      % proccessing the mask volume
      h_new = int16( alpha*height );
	  w_new = int16( alpha*width );
	  n_new = int16( beta*number );              
      allMaxRadiologistMsk_new = zeros(h_new,w_new,n_new);
      for n=1:n_new
        n_old = min(number,int16(n/beta));        
        %currentSlice = allMaxRadiologistMsk(:,:,n_old);
        %figure,imshow(currentSlice);
        for h=1:h_new
          h_old = min(height,int16(h/alpha));
          for w=1:w_new
            w_old = min(width,int16(w/alpha));
            allMaxRadiologistMsk_new(h,w,n) = allMaxRadiologistMsk(h_old,w_old,n_old);
          end
        end
        %close all;
      end
      allMaxRadiologistMsk = allMaxRadiologistMsk_new;
      
      % proccesing the uniqueStats
      numFields = numel(fieldnames(uniqueStats));
      if numFields == 27
          [numTags,~]=size(uniqueStats);
          for n=1:numTags
            uniqueStats(n).CasePath = uniqueStats(n).CasePath;
            uniqueStats(n).Tag = uniqueStats(n).Tag;
            uniqueStats(n).NumberOfObservers = uniqueStats(n).NumberOfObservers;
            uniqueStats(n).ReadersIDs = uniqueStats(n).ReadersIDs;
            uniqueStats(n).IDs = uniqueStats(n).IDs;
            uniqueStats(n).IURatio = uniqueStats(n).IURatio;
            uniqueStats(n).avgVolume = uniqueStats(n).avgVolume;
            uniqueStats(n).avgPrincipalAxisLength = uniqueStats(n).avgPrincipalAxisLength;
            uniqueStats(n).PixelSpacing = uniqueStats(n).PixelSpacing;
            uniqueStats(n).SliceThickness = uniqueStats(n).SliceThickness;
            uniqueStats(n).SliceThicknessDicom = uniqueStats(n).SliceThicknessDicom;
            uniqueStats(n).avgSubtlety = uniqueStats(n).avgSubtlety;
            uniqueStats(n).avgCalcification = uniqueStats(n).avgCalcification;
            uniqueStats(n).avgSphericity = uniqueStats(n).avgSphericity;
            uniqueStats(n).avgMargin = uniqueStats(n).avgMargin;
            uniqueStats(n).avgLobulation = uniqueStats(n).avgLobulation;
            uniqueStats(n).avgSpiculation = uniqueStats(n).avgSpiculation;
            uniqueStats(n).avgMalignancy = uniqueStats(n).avgMalignancy;
            
            uniqueStats(n).minX = max(1,int16(uniqueStats(n).minX/alpha)); % x is w
            uniqueStats(n).maxX = min(int16(uniqueStats(n).maxX/alpha),width);
            uniqueStats(n).minY = max(1,int16(uniqueStats(n).minY/alpha)); % y is h
            uniqueStats(n).maxY = min(int16(uniqueStats(n).maxY/alpha),height);
            uniqueStats(n).minZ = max(1,int16(uniqueStats(n).minZ/beta)); % z is n
            uniqueStats(n).maxZ = min(int16(uniqueStats(n).maxZ/beta),n_new);
            uniqueStats(n).avgCentroidX = int16(uniqueStats(n).avgCentroidX/alpha);
            uniqueStats(n).avgCentroidY = int16(uniqueStats(n).avgCentroidY/alpha);
            uniqueStats(n).avgCentroidZ = int16(uniqueStats(n).avgCentroidZ/beta);
            minSlice = max(1,int16(uniqueStats(n).minSlice/beta));
            uniqueStats(n).minSlice = minSlice;
            maxSlice = min(int16(uniqueStats(n).maxSlice/beta),n_new);
            uniqueStats(n).maxSlice = maxSlice;
            uniqueStats(n).avgVolCentroidX = int16(uniqueStats(n).avgVolCentroidX/alpha);
            uniqueStats(n).avgVolCentroidY = int16(uniqueStats(n).avgVolCentroidY/alpha); 
            uniqueStats(n).avgVolCentroidZ = int16(uniqueStats(n).avgVolCentroidZ/beta);
            uniqueStats(n).avgSliceCentroidX = int16(uniqueStats(n).avgSliceCentroidX/alpha);
            uniqueStats(n).avgSliceCentroidY = int16(uniqueStats(n).avgSliceCentroidY/alpha);
            uniqueStats(n).avgMsk = allMaxRadiologistMsk(:,:,minSlice:maxSlice);
          end
      end
      savedir=strcat(savepath,matname);
      save(savedir, 'allMaxRadiologistMsk','listDicomInfo','uniqueStats');
      cd ..      
    end
    cd ..
  end
  cd ..
end