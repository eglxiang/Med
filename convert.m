## Copyright (C) 2017 Xiang Xiang
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} convert (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Xiang Xiang <eglxiang@gmail.com>
## Created: 2017-07-12

alpha = 0.80
beta = 0.70
rootpath = '/raida/apezeshk/lung_dicom_dir/'
savepath = '/media/shamidian/sda2/ws/mat_resized/'

cd(rootpath);
exterior_folders=dir;
[num_exterior_folders,~]=size(exterior_folders);
for i=5:num_exterior_folders
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
      filename=ls('*.mat');
      load(filename);
      [height,width,number]=size(allMaxRadiologistMsk);
      h_new = int8( alpha*height )
	    w_new = int8( alpha*width )
	    n_new = int8( beta*number )              
      newVol = zeros(h_new,w_new,n_new);
      for n=1:n_new
        n_old = int8(n/beta);
        #currentSlice = allMaxRadiologistMsk(:,:,n_old);
        #figure,imshow(currentSlice);
        for h=1:h_new
          h_old = int8(h/alpha);
          for w=1:w_new
            w_old = int8(w/alpha);
            newVol(h,w,n) = allMaxRadiologistMsk(h_old,w_old,n_old);
          end
        end
        #close all;
      end
      savename=strcat(current_exterior_folder,'_',current_sub_folder,'_',current_interior_folder,'.mat');
      savedir=strcat(savepath,savename);
      save(savedir, 'newVol');
      cd ..      
    end
    cd ..
  end
  cd ..
end