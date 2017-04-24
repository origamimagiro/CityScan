%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The MIT License
% 
% Copyright (c) 2016 Jason S. Ku
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TO DO
% [ ] - Separate plotting from testing code
% [ ] - Fix import_ios noise filter
% [ ] - Add test functions and update header
%
% 2016-07-25
% [x] - Added some daytime image library functionality
%
% 2016-07-24
% [x] - Added function to estimate lamp power
% [x] - Added function to return lane widths
% [x] - Added video writing for many test functions
%
% 2016-07-23
% [x] - Added function to return angles from x, y
%
% 2016-07-22
% [x] - Added function to get centroids from a frame
% [x] - Added function to get all centroids from a movie
% [x] - Added function to convert a centroid to an observation
% [x] - Added function to import observation from a movie
% [x] - Improved GPS importer
%
% 2016-07-18
% [x] - Optimization: halved running time of library importer
% [x] - Generalized plot_lib to be localized at a specified location
%
% 2016-07-17
% [x] - Added location library file name function
% [x] - Added a function to draw an image onto another image
% [x] - Added a frame to the library importer
% [x] - Added a movie to a library importer
% [x] - Added library plotting function
%
% 2016-07-15
% [x] - Added world pixel coordinate alignment to perspective_transform
% [x] - Cleaned up image scaling/removed costly projective imwarp
%
% 2016-07-14
% [x] - Reorganized method organization
% [x] - Again rewrote JSON parser
% [x] - Rewrote data import/save methods
% [x] - Restructred landmark observation data structure
% [x] - Cleaned up perspective_transform code with new tests
%
% 2016-07-13
% [x] - Researched regression path models for vehicle trajectory
% [x] - Converted interpolation to smoothing spline regression
%       https://en.wikipedia.org/wiki/Smoothing_spline
%       http://www.mathworks.com/help/curvefit/smoothing-splines.html
% [x] - Incorporated into iOS data import
%
% 2016-07-12
% [x] - Reorganized existing functions 
% [x] - Cleaned up data import to structure format
% [x] - Rewrote JSON parser
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CITYSCAN LIBRARY
% MIT & FERROVIAL
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef CityScan
methods(Static = true)

function [] = test_all()
  fns = {'check_file_existence',    ... % File Utilities 
         'get_string_from_file',    ...
         'files_in_path_with_exts', ...
         'get_trial',               ... % Data Import
         'save_mat_from_trial',     ...
         'import_ios',              ...
         'import_lp',               ...
         'import_mov',              ...
         'parse_json',              ... 
         'partition_obs',           ... % Landmark Localization
         'estimate_position',       ... 
         'min_segment',             ...
         'vanishing_point',         ... % Ground Projection
         'get_vanishing_points',    ...
         'perspective_transform',   ...
         'place_image_on_image',    ...
         'name_from_location',      ...
         'lib_from_frame',     ...
         'import_lib_from_mov',     ...
         'plot_lib'};
  for i = 1:numel(fns)
    fname = ['test_' fns{i}];
    disp('*******************************');
    disp(['Running ' fname '() ...']);
    disp('*******************************');
    eval(['CityScan.' fname]);
  end
end


%---------------------%
%    File Utilities   % 
%---------------------%
% [ exists ] = check_file_existence( filename, flag )
% [ string ] = get_string_from_file( filename )
% [ files ]  = files_in_path_with_exts( path, exts )


function [ exists ] = check_file_existence( filename, flag )
  exists = exist(filename, 'file') == 2;
  if flag
    assert(exists, ['ERROR: Could not find file ' filename]);
  end
end
function [] = test_check_file_existence()
  filename = 'ex/lp/test1.mov';
  disp(['Testing for the existence of file: ' filename]);
  if CityScan.check_file_existence(filename,false)
    disp([filename ' exists!']);
    filename = 'ex/lp/aajfdfahkjhlfa.mov';
    disp(['Now testing for the existence of file: ' filename]);
    try
      CityScan.check_file_existence(filename,true);
    catch
      disp(['We caught an exception! Looks like' filename ' does not exist.']);
    end
  end
end

function [ string ] = get_string_from_file( filename )
  CityScan.check_file_existence(filename, true);
  file = fopen(filename);
  string = fread(file, '*char')';
  fclose(file);
end
function [] = test_get_string_from_file()
  filename = 'ex/lp/test1.txt';
  string = CityScan.get_string_from_file(filename);
  disp(['File read: ' filename]);
  disp(['Number of characters: ' num2str(numel(string))]);
  disp('First 100 characters:');
  disp(string(1:100));
end

function [ files ] = files_in_path_with_exts( path, exts )
  data = dir(path);
  allfiles = struct;
  files = {};
  for i = 1:numel(data)
    split = strsplit(data(i).name,'.');
    if (numel(split) == 2) && ~strcmp(split{1},'')
      if ~isfield(allfiles,split{1}) 
        allfiles.(split{1}) = {};
      end
      allfiles.(split{1}) = [allfiles.(split{1}), split{2}];
    end  
  end
  names = fieldnames(allfiles);
  for i = 1:numel(names)
    if all(ismember(exts,allfiles.(names{i})))
      files = [files,names{i}];
    end
  end
end
function [] = test_files_in_path_with_exts()
  path = 'ex/night_side/'; 
  exts = {'txt','mov','lp'};
  disp(['Files in path ' path ' with extensions']); 
  disp(exts);
  disp(CityScan.files_in_path_with_exts(path,exts)');
end


%-----------------%
%   Data Import   %
%-----------------%
% [ trial ] = get_trial( filename )
%        [] = save_mat_from_trial( filename )
%   [ ios ] = import_ios( filename )
%    [ lp ] = import_lp( filename, verbose )
%   [ mov ] = import_mov( filename )
%   [ obj ] = parse_json( string, s, e, verbose )

function [ trial ] = get_trial( filename )
  if ~CityScan.check_file_existence([filename '.mat'],false)
    CityScan.save_mat_from_trial(filename); 
  end
  trial = load([filename '.mat']);
end
function [] = test_get_trial()
  filename  = 'ex/day_front/road1';
  disp(['Getting trial at : ' filename '.mat ...']);
  trial = CityScan.get_trial(filename)
end

function [] = save_mat_from_trial( filename )
  exts = {'.txt','.lp','.mov'};
  mov = struct; ios = struct; lp  = struct; 
  for i = 1:3
    file = [filename exts{i}];
    if CityScan.check_file_existence(file,false)
      disp([exts{i} ' data found! Processing...']);
      switch i
        case 1; ios = CityScan.import_ios(file);
        case 2;  lp = CityScan.import_lp(file,false);
        case 3; mov = CityScan.import_mov(file);
      end 
    end
  end 
  disp(['Saving trial data to file: ' filename '.mat']);
  save([filename '.mat'],'ios','lp','mov');
end
function [] = test_save_mat_from_trial()
  filename  = 'ex/day_front/road1';
  disp(['Importing data from files with path: ' filename]);
  CityScan.save_mat_from_trial(filename);
end

function [ mov ] = import_mov( filename )
  CityScan.check_file_existence(filename,true);
  disp(['File read: ' filename]);
  disp('Counting movie frames...');
  vid = VideoReader(filename);
  n = vid.NumberOfFrames;
  disp(['Video has ' num2str(n) ' frames.']);
  times = (0:(n - 1)) / vid.FrameRate;
  f = 2961;
  mov = struct('n',n,'times',times,'f',f);
end
function [] = test_import_mov()
  filename  = 'ex/day_front/road1.mov';
  CityScan.import_mov(filename);
end

function [ ios ] = import_ios( filename )
  CityScan.check_file_existence(filename,true);
  disp(['File read: ' filename]);
  disp('Cleaning GPS data...');
  ios = struct;
  warning('off', 'MATLAB:table:ModifiedVarnames');
  data = table2struct(readtable(filename));
  names = fieldnames(data);
  for i = 1:numel(names)
    ios(1).(names{i}) = [data.(names{i})]';
  end
  assert(all(isfield(data,{'time','lon','lat'})), ...
    'Error: File does not contain TIME, LAT, and LON data');
  unique = ios.lat ~= circshift(ios.lat,[-1,0]);
  unique = mod(1:numel(ios.lat),30) == 1;
  ios.time = ios.time - min(ios.time);
  ios.time_s = ios.time(unique);
  t = ios.time_s - circshift(ios.time_s,[1,0]);
  ios.lat_s = ios.lat(unique);
  ios.lon_s = ios.lon(unique);
  EARTH_RADIUS_METERS = 6378137;
  ios.lat_m = EARTH_RADIUS_METERS*sind(ios.lat_s);
  ios.lon_m = EARTH_RADIUS_METERS*cosd(mean(ios.lat_s))*sind(ios.lon_s);
  u = [0,0]; dist = 0; i = 1;
  % min_rad = 2; % vehicle turning radius lower bound
  % max_dist = 50;
  % while dist < min_rad
  %   i = i + 1;
  %   u = [ios.lat_m(i) - ios.lat_m(1); ...
  %        ios.lon_m(i) - ios.lon_m(1)];
  %   dist = norm(u);
  % end
  % u = unit(u);
  % for i = 2:numel(ios.time_s)
  %   new_u = [ios.lat_m(i) - ios.lat_m(i - 1); ...
  %            ios.lon_m(i) - ios.lon_m(i - 1)];
  %   dist = norm(new_u); 
  %   if dist == 0; continue; end
  %   ang = acos(unit(new_u)' * u); 
  %   if ang * min_rad > dist || dist > max_dist
  %     dir = u * max(0, u' * new_u);
  %     ios.lat_m(i) = ios.lat_m(i - 1);
  %     ios.lon_m(i) = ios.lon_m(i - 1);
  %   else
  %     u = unit(new_u);
  %   end
  % end
  fo = fitoptions('smoothingspline');
  fo.SmoothingParam = 0.2;
  ios.f_lon_m = fit(ios.time_s,ios.lon_m,'smoothingspline',fo);
  ios.f_lat_m = fit(ios.time_s,ios.lat_m,'smoothingspline',fo);
end
function [] = test_import_ios()
  filename = 'ex/day_side/day_120cm_90deg.txt';
  ios = CityScan.import_ios(filename);
  figure; hold on;
  minlon = min(ios.lon_m); minlat = min(ios.lat_m);
  plot(ios.lon_m - minlon,ios.lat_m - minlat,'ro-');
  n = numel(ios.time) * 10;
  range = max(ios.time_s) - min(ios.time_s);
  t = (0:n)/n*range + min(ios.time_s);
  plot(ios.f_lon_m(t) - minlon,ios.f_lat_m(t) - minlat,'k-');
  axis equal;
  % figure;
  % plot(t,ios.f_lon_m(t) - minlon,'b-'); hold on;
  % plot(ios.time_s,ios.lon_m - minlon,'ro-');
  % plot(t,ios.f_lat_m(t) - minlat,'b-');
  % plot(ios.time_s,ios.lat_m - minlat,'ro-');
end

function [ lp ] = import_lp( filename, verbose )
  string = CityScan.get_string_from_file(filename);
  disp(['File read: ' filename]);
  disp('Parsing JSON...');
  lp = CityScan.parse_json(string,1,numel(string),verbose);
  disp('Cleaning spectrum data points...')
  for i = 1:numel(lp.data_points)
    if verbose
      disp([sprintf('% 6.2f',i/numel(lp.data_points)*100) ' %']);
    end
    sp = lp.data_points{i}.spectrumPoints;
    sp_ = struct;
    sp_(1).wavelength = zeros(numel(sp),1);
    sp_.rel_intensity = zeros(numel(sp),1);
    for j = 1:numel(sp)
      wl = fieldnames(sp{j});
      sp_.wavelength(j) = str2num(wl{1}(2:end));
      sp_.rel_intensity(j) = sp{j}.(wl{1});
    end
    lp.data_points{i}.spectrumPoints = sp_;
  end
  data = [lp.data_points{:}];
  time = [data.timestamp];
  [~, sort_idx] = sort(time);
  lp.data_points = data(sort_idx);
end
function [] = test_import_lp()
  filename  = 'ex/lp/test1.lp';
  tic; json = CityScan.import_lp(filename, false); toc;
end

function [ obj ] = parse_json( string, s, e, verbose )
  switch string(s)
    case '{';  obj = struct;
    case '[';  obj = {};
    otherwise
      if verbose
        disp([sprintf('% 6.2f',s/numel(string)*100) ' %']);
      end
      switch string(s)
        case 'n';  obj = [];
        case 't';  obj = true;
        case 'f';  obj = false;
        case '"';  obj = string((s + 1):(e - 1));
        otherwise; obj = str2num(string(s:e));
      end
      return
  end
  depth = 0; start = s + 1;
  for i = (s + 1):e
    add = false;
    switch string(i)
      case {'{','['}; depth = depth + 1; 
      case {'}',']'}; depth = depth - 1;
      case ':'; if depth == 0; key = i; end;
      case ','; if depth == 0; add = true; end;
    end
    if add || i == e
      switch string(s)
        case '{'
          field = string((start + 1):(key - 2));
          if ~isletter(field(1)); field = ['f' field]; end
          obj.(field) = CityScan.parse_json(string,key + 1,i - 1,verbose);
        case '['
          obj = [obj,  {CityScan.parse_json(string,  start,i - 1,verbose)}];
      end
      start = i + 1;
    end
  end
end
function [] = test_parse_json()
  filename  = 'ex/lp/test1.lp';
  string = CityScan.get_string_from_file(filename);
  disp(['File read: ' filename]);
  tic; CityScan.parse_json(string,1,numel(string),false); toc;
end


%---------------------------%
%   Landmark Localization   %
%---------------------------%
% FORM      obs = struct('pos',{[0;0;0]},'dir',{[1;0;0]})
% [ centroids ] = centroids_in_frame( I, t )
% [ centroids ] = get_centroids( filename, n, replace )
%       [ obs ] = obs_from_centroid( centroid, f, p, t )
%       [ obs ] = import_obs_from_mov( filename, n )
%     [ label ] = partition_obs( obs, t, r )
%     [ p, s ]  = estimate_position( obs, sp, st )
%     [ a, b ]  = min_segment( obs1, obs2 )


function [ centroids ] = centroids_in_frame( I, t )
  centroids = {};
  regions = regionprops(bwpropfilt( ...
    im2bw(I),'Area',[50 400].^2,8), 'Centroid');
  if numel(regions) == 0; return; end
  for i = 1:numel(regions)
    centroids{i} = regions(i).Centroid - fliplr(size(I(:,:,1))) / 2;
  end
end
function [] = test_centroids_in_frame()
  filename = 'ex/night_side/night_120cm_90deg_1.mov';
  disp(['Using Example: ' filename]);
  CityScan.check_file_existence(filename,true);
  vid = VideoReader(filename);
  for j = 1:550
    I = readFrame(vid);
    if j > 400
      cla;
      I = flipud(permute(I,[2,1,3]));
      centroids = CityScan.centroids_in_frame(I,0.98);
      imshow(I);
      axis on;
      for i = 1:numel(centroids)
        c = centroids{i};
        hold on; 
        plot(c(1)+1920 / 2,c(2) + 1080 / 2,'ro');
      end
      title(num2str(j));
      drawnow;
    end
  end
end

function [ centroids ] = get_centroids( filename, n, replace )
  disp(['Using Example: ' filename]);
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  if n ~= 0 && ~replace && isfield(trial.mov,'centroids') && ... 
       numel(trial.mov.centroids) >= n
    centroids = trial.mov.centroids; return;
  end
  CityScan.check_file_existence([filename '.mov'],true);
  vid = VideoReader([filename '.mov']);
  centroids = {}; i = 1; tic;
  while hasFrame(vid)
    if mod(i,10) == 0
      disp(['Processing frame #' num2str(i)]);
    end 
    I = flipud(permute(readFrame(vid),[2,1,3]));
    centroids{i} = CityScan.centroids_in_frame(I,0.98);
    if i == n; break; end
    i = i + 1;
  end; toc;
  mov.centroids = centroids;
  disp(['Saving centroids to file: ' filename '.mat']);
  save([filename '.mat'],'ios','lp','mov');
end
function [] = test_get_centroids()
  filename = 'ex/night_side/night_120cm_90deg_1';
  centroids = CityScan.get_centroids(filename,10,false);
  trial = CityScan.get_trial(filename);
  fig = figure;
  set(fig,'Color','w');
  n = 5003;
  start_time = trial.mov.times(n);
  vid = VideoReader([filename '.mov']);
  vid.currentTime = start_time;
  % out = VideoWriter('centroids','MPEG-4');
  % out.FrameRate = 10;
  % open(out); 
  for i = (n - 3):(n + 500 - 3)
    I = flipud(permute(readFrame(vid),[2,1,3]));
    cla;
    imshow(I); hold on;
    fcentroid = centroids{i};
    for j = 1:numel(fcentroid)
      plot(fcentroid{j}(1) + 1920 / 2,fcentroid{j}(2) + 1080 / 2,'ro', ...
        'MarkerSize',40,'LineWidth',8);
    end
    title(num2str(i - 5000));
    drawnow;
    % img = frame2im(getframe(fig));
    % writeVideo(out,img);
  end
  % close(out);
end

function [ ang ] = ang_from_xy( x, y )
  min_rad = 2;
  assert(numel(x) > 1, 'Error: cannot return angle from one data point');
  ang = x * 0;
  for i = 1:numel(x)
    j = i + 1; dist = 0;
    while (dist < min_rad) && (j < numel(x))
      dist = norm([y(j) - y(i);x(j) - x(i)]);
      j = j + 1;
    end
    if j >= numel(x)
      ang(i) = ang(i - 1); 
    else
      ang(i) = atan2(y(j)-y(i),x(j)-x(i));
    end
  end
end

function [ obs ] = obs_from_centroid( centroid, f, p, t )
  obs = struct('pos',{},'dir',{}); 
  R1 = [cos(t(3)), sin(t(3)), 0; -sin(t(3)), cos(t(3)), 0; 0, 0, 1];
  R2 = [1, 0, 0; 0, cos(t(2)), -sin(t(2)); 0, sin(t(2)), cos(t(2))];
  R3 = [cos(t(1)), -sin(t(1)), 0; sin(t(1)), cos(t(1)), 0; 0, 0, 1];
  u = unit([-centroid(1);-centroid(2);f]);
  obs(1).pos = p;
  obs.dir = R3 * R2 * R1 * u;
end
function [] = test_obs_from_centroid()
  filename = 'ex/night_side/night_120cm_90deg_1';
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  n = 1000; fig = figure;
  set(gcf,'Color','w','Position',[10,10,800,800]);
  centroids = CityScan.get_centroids(filename,n,false);
  x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  x = x - min(x); y = y - min(y);
  ang = CityScan.ang_from_xy(x,y);
  % out = VideoWriter('camera','MPEG-4');
  % out.FrameRate = 30;
  % open(out); 
  for i = 5000:5500; 
    p = [x(i);y(i);1];
    t = [ang(i),0,0];
    % subplot(1,2,1); cla;
    % for j = 1:numel(centroids{i})
    %   plot(centroids{i}{j}(1),-centroids{i}{j}(2),'ro');
    % end
    % title(num2str(i)); axis equal;
    % set(gca,'YLim',[-1,1] * 1080 / 2, ...
    %         'XLim',[-1,1] * 1920 / 2);
    % subplot(1,2,2); cla;
    cla;
    r = max(1,i-500):min(i+500,numel(centroids));
    plot3(x(r),y(r),x(r) * 0,'k-','LineWidth',2); hold on;
    o = CityScan.obs_from_centroid([0,0],mov.f,p,t);
    array = [1,1;1,-1;-1,-1;-1,1;1,1];
    m = size(array,1); ps = zeros(m,2,3);
    array = array .* (ones(m,1) * [1920,1080] / 2);
    for j = 1:m
      obs = CityScan.obs_from_centroid(array(j,:),mov.f,p,t);
      s = obs.dir' * o.dir;
      ps(j,:,:) = [obs.pos,obs.pos + obs.dir / s]';
    end
    plot3(ps(:,:,1)',ps(:,:,2)',ps(:,:,3)','b-','LineWidth',2);
    plot3(ps(:,:,1) ,ps(:,:,2) ,ps(:,:,3) ,'b-','LineWidth',2);
    plot3(ps(1,1,1),ps(1,1,2),0,'bo','LineWidth',2);
    for j = 1:numel(centroids{i})
      obs = CityScan.obs_from_centroid(centroids{i}{j},mov.f,p,t);
      s = obs.dir' * o.dir;
      ps = [obs.pos,obs.pos + obs.dir / s];
      plot3(ps(1,:),ps(2,:),ps(3,:),'r-o','LineWidth',2);
    end
    title(num2str(i - 5000 + 1)); axis equal;
    grid on; 
    set(gca,'ZGrid','off','LineWidth',2);
    set(gca,'XTickLabel','','YTickLabel','','ZTickLabel','');
    set(gca,'Position',[0.05,0.05,0.9,0.9]);
    xl = round(get(gca,'XLim')); yl = round(get(gca,'YLim'));
    set(gca,'XTick',xl(1):xl(2),'YTick',yl(1):yl(2));
    set(gca,'XLim',[x(i)-2,x(i)+2],'YLim',[y(i)-2,y(i)+2],'Zlim',[0,2]);
    if mod(i,1) == 0; drawnow; end
    % img = frame2im(getframe(fig));
    % writeVideo(out,img);
  end
  % close(out);
end

function [] = import_obs_from_mov( filename, n, replace )
  disp(['Using Example: ' filename]);
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  if ~replace && (n ~= 0) && isfield(mov,'centroids') && ...
      (numel(mov.centroids) >= n)
    return;
  end;
  CityScan.check_file_existence([filename '.mov'],true);
  vid = VideoReader([filename '.mov']);
  disp('Calculating centroids from frames...');
  m = 0;
  while hasFrame(vid)
    m = m + 1;
    if mod(m,10) == 0; disp(['Processing frame #' num2str(m)]); end 
    I = flipud(permute(readFrame(vid),[2,1,3]));
    mov.centroids{m} = CityScan.centroids_in_frame(I,0.99);
    if m == n; break; end
  end;
  disp('Calculating observations from centroids...');
  x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  ang = CityScan.ang_from_xy(x,y);
  mov.obs = {};
  for i = 1:m
    p = [x(i);y(i);1]; t = [ang(i),0,0];
    for j = 1:numel(mov.centroids{i})
      ob = CityScan.obs_from_centroid(mov.centroids{i}{j},mov.f,p,t);
      ob.time = mov.times(i);
      mov.obs = [mov.obs {ob}];
    end
  end
  disp(['Saving centroids and observations to file: ' filename '.mat']);
  save([filename '.mat'],'ios','lp','mov');
end
function [] = test_import_obs_from_mov()
  filename = 'ex/night_side/night_120cm_90deg_1';
  n = 10;
  CityScan.import_obs_from_mov(filename,n,false);
  trial = CityScan.get_trial(filename);
  fig = figure;
  set(fig,'Color','w','Position',[10,10,800,800]);
  obs = trial.mov.obs;
  obs_set = {}; sn = 1;
  % out = VideoWriter('locate','MPEG-4');
  % out.FrameRate = 30;
  % open(out); 
  for i = 1:numel(obs); 
    ps = [obs{i}.pos,obs{i}.pos + obs{i}.dir * 30];
    hold on;
    plot3(ps(1,:),ps(2,:),ps(3,:),'r-','LineWidth',1);
    if i == numel(obs) || (norm(obs{i}.pos - obs{i+1}.pos) > 3)
      [p,s] = CityScan.estimate_position(obs_set,eye(3)*0.1,eye(3)*0.1);
      axis tight;
      plot3(trial.ios.f_lon_m(trial.mov.times), ...
            trial.ios.f_lat_m(trial.mov.times),trial.mov.times * 0,'b-', ...
            'LineWidth',2);
      plot3(p(1),p(2),p(3),'ko','LineWidth',8,'MarkerSize',30); 
      axis equal;
      s = 10;
      xlim([p(1) - s,p(1) + s]);
      ylim([p(2) - s,p(2) + s]);
      zlim([0,25]);
      grid on;
      set(gca,'Position',[0.05,0.05,0.9,0.9]);
      set(gca,'XTickLabel','','YTickLabel','','LineWidth',2);
      set(gca,'FontSize',20);
      %title(num2str(obs{i}.time));
      if sn == 2
        for j = 1:200
          set(gca,'CameraPosition', ...
            [p(1) + 12 * cos(pi * 0.8 + pi * j / 100), ...
             p(2) + 12 * sin(pi * 0.8 + pi * j / 100), ...
             p(3) - 3],'CameraViewAngle',100);
            drawnow;
          % img = frame2im(getframe(fig));
          % writeVideo(out,img);
        end
        % close(out);
        return;
      end
      cla;
      obs_set = {}; sn = sn + 1;
    else
      obs_set = [obs_set obs(i)];
    end
  end
end

function [ lamps ] = add_lamp_images_from_trial( lamps, filename )
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  for i = 1:numel(lamps)
    min_dist = 500; idx = 0;
    for j = 1:numel(x)
      dist = norm(lamps{i}.p - [x(j);y(j);1]);
      if dist < min_dist
        min_dist = dist;
        idx = j; 
      end
    end 
    if idx ~= 0
      lamps{i}.temp_idx = idx; 
      disp([num2str(i) ' min_dist:' num2str(min_dist)]);
    end
  end
  vid = VideoReader([filename '.mov']); i = 1;
  while hasFrame(vid)
    if mod(i,10) == 0; disp(['Processing frame #' num2str(i)]); end
    lamp_idx = 0; 
    for j = 1:numel(lamps)
      if isfield(lamps{j},'temp_idx') && (lamps{j}.temp_idx == i)
        lamp_idx = j; 
        break; 
      end
    end 
    if lamp_idx == 0
      readFrame(vid);
    else
      lamps{lamp_idx}.img = readFrame(vid); 
    end
    i = i + 1;
  end
end
function [ lamps ] = test_add_lamp_images_from_trial()
  % Requires trial.mov.obs to have already been populated
  filename = 'ex/night_side/night_120cm_90deg_1';
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  fig = figure;
  set(fig,'Color','w','Position',[10,10,800,600]);
  lamps = CityScan.get_lamps_from_obs(mov.obs);
  x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  % plot3(x,y,x * 0,'b-','LineWidth',2); hold on;
  hold on;
  %plot(x,y,'r');
  day = 'ex/day_side/day_120cm_67deg';
  % trial = CityScan.get_trial(day);
  % ios = trial.ios; lp = trial.lp; mov = trial.mov;
  % x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  % plot(ios.lon_m,ios.lat_m,'bo');
  lamps = CityScan.add_lamp_images_from_trial(lamps,day);
  for i = 1:numel(lamps); if isfield(lamps{i},'img')
    cla;
    imshow(lamps{i}.img);
    drawnow;
    pause;
  end; end
end

function [ lamps ] = get_lamps_from_obs( obs )
  lamps = {};
  obs_set = {};
  min_sep = 3; max_sight = 30;
  for i = 1:numel(obs); 
    ps = [obs{i}.pos,obs{i}.pos + obs{i}.dir * max_sight];
    if i == numel(obs) || (norm(obs{i}.pos - obs{i+1}.pos) > min_sep)
      [p,s] = CityScan.estimate_position(obs_set,eye(3)*0.1,eye(3)*0.1);
      lamps = [lamps, {struct('p',p,'s',s)}]; 
      obs_set = {};
    else
      obs_set = [obs_set obs(i)];
    end
  end
end
function [] = test_get_lamps_from_obs()
  filename = 'ex/night_side/night_120cm_90deg_1';
  % n = 0;
  % CityScan.import_obs_from_mov(filename,n,true);
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  fig = figure;
  set(fig,'Color','w','Position',[10,10,1000,500]);
  lamps = CityScan.get_lamps_from_obs(trial.mov.obs);
  x = ios.f_lon_m(mov.times); y = ios.f_lat_m(mov.times);
  plot3(x,y,x * 0,'b-','LineWidth',2); hold on;
  for i = 1:numel(lamps); 
    p = lamps{i}.p;
    plot3([p(1),p(1)],[p(2),p(2)],[0,p(3)],'ro-','LineWidth',2);
  end
  lightfile = 'ex/lp/test6';
  light = CityScan.get_trial(lightfile);
  ios = light.ios; lp = light.lp; mov = light.mov;
  t = [lp.data_points.timestamp];
  t = t - min(t);
  xt = ios.f_lon_m(t);
  yt = ios.f_lat_m(t);
  lux = [lp.data_points.lux];
  lux = lux' / max(lux) * 50;
  hold on;
  lamps = CityScan.estimate_lamp_power(lamps,xt,yt,ones(size(xt)),lux);
  plot3([xt,xt],[yt,yt],[lux,zeros(size(lux))],'g-','LineWidth',2);
  plot3([xt,xt]',[yt,yt]',[lux,zeros(size(lux))]','g-');
  li = 33;
  for i = (li - 2):(li + 2)
    l = lamps{i};
    t = text(l.p(1),l.p(2),l.p(3),num2str(round(l.power / 10)));
    off = 4;
    t.Position = [l.p(1),l.p(2),l.p(3) + off];
    t.FontSize = 15;
  end
  axis equal;
  zlim([0,25]);
  xlim([lamps{li}.p(1) - 70, lamps{li}.p(1) + 70]);
  ylim([lamps{li}.p(2) - 70, lamps{li}.p(2) + 70]);
  grid on;
  set(gca,'XTickLabel','','YTickLabel','', ...
    'ZTick',[0,5,10,15,20,25]);
  out = VideoWriter('power','MPEG-4');
  out.FrameRate = 30;
  open(out); 
  for i = 1:200
    set(gca,'CameraPosition', ...
      [lamps{li}.p(1) + 80 * cos(pi * 1.8 + pi * i / 100), ...
       lamps{li}.p(2) + 80 * sin(pi * 1.8 + pi * i / 100),30], ...
      'CameraViewAngle',65,'Position',[0.05,0.05,0.9,0.9], ...
      'FontSize',15);
    drawnow;
    img = frame2im(getframe(fig));
    writeVideo(out,img);
  end
  close(out);
end

function [ lamps ] = estimate_lamp_power( lamps, x, y, z, lux )
  max_dist = 50;
  A = zeros(numel(lamps),numel(lux));
  for i = 1:numel(lamps)
    for j = 1:numel(lux)
      dist = norm(lamps{i}.p - [x(j);y(j);z(j)]);
      if dist < max_dist; A(i,j) = 1 / dist^2; end
    end
  end
  %figure;
  %spy(A)
  %pause; 
  power = A' \ lux;
  for i = 1:numel(lamps)
    lamps{i}.power = power(i);
  end
end

function [ label ] = partition_obs( obs, t, r )
% PARTITION_OBS returns an array of labels associated with each observation such
% that two have the same label when the distance between them is less than
% threshold t and the midpoint of the shortest segment between them is within
% distance r from the observation point in the observation direction.
% [ label ] = partition_obs( obs, t, r );
  n = numel(obs);
  label = zeros(n,1);
  new_label = 0;
  for i = 1:n
    if label(i) == 0
      new_label = new_label + 1;
      label(i)  = new_label;
    end
    for j = (i+1):n
      [a, b] = CityScan.min_segment(obs(i),obs(j));
      m = (a + b) / 2;
      tij = norm(a - b);
      ri = dot(m - obs(i).pos, obs(i).dir) / norm(obs(i).dir);
      rj = dot(m - obs(j).pos, obs(j).dir) / norm(obs(j).dir);
      if (tij < t) && (ri > 0) && (ri < r) && (rj > 0) && (rj < r)
        label(j) = label(i);
      end
    end
  end
end
function [] = test_partition_obs()
  obs = struct('pos',{[0;0;0],[0;1;-0.5],[0;-1;-0.1]}, ...
               'dir',{[1;0;1],  [1;-1;1],   [1;1,;1]});
  r = 2; t = [0.01,0.3,0.4];
  linespec = {'b-o','r-o','g-o'};
  figure;
  for i = 1:numel(t)
    label = CityScan.partition_obs(obs,t(i),r);
    subplot(1,3,i); hold on;
    for j = 1:numel(obs)
      ps = obs(j).pos;
      qs = ps + obs(j).dir * r / norm(obs(j).dir);
      plot3([ps(1),qs(1)], ...
            [ps(2),qs(2)], ...
            [ps(3),qs(3)],linespec{label(j)});
    end
    axis equal;
  end
end

function [ p, s, A, B, b ] = estimate_position( obs, sp, st )
% ESTIMATE_POSITION returns position p and covariance matrix s consistent with 
% observations provided, given position and direction covariances sp and st.
% [ p, s ] = estimate_position( obs, sp, st, A, b, B )
  A = zeros(3);
  B = zeros(3);
  b = zeros(3,1);
  for i = 1:numel(obs)
    p_ = obs{i}.pos;
    u_ = obs{i}.dir;
    u_ = u_ / norm(u_);
    P_ = (eye(3) - u_*u_');
    A = A + P_; 
    b = b + P_*p_;
    B = B + P_*sp*P_ + p_'*st*p_;
  end
  p = A\b;
  s = A*B*A;
end
function [] = test_estimate_position()
  p = [1;1;1]; n = 10;
  var_pos = 0.1; var_dir = 0.01;
  x = (0:n) / (n - 1) * 2;
  obs = struct;
  for i = 1:n
    obs(i).pos = [x(i);0;0] + var_pos * randn(3,1);
    obs(i).dir =  unit(p - obs(i).pos) + var_dir * randn(3,1);
  end
  [q,s] = CityScan.estimate_position(obs,var_pos * eye(3),var_dir * eye(3));
  figure; hold on;
  plot3(q(1),q(2),q(3),'ob','LineWidth',2);
  plot3(p(1),p(2),p(3),'or','LineWidth',2);
  plot3(x,x * 0,x * 0,'ro-','LineWidth',2);
  for i = 1:numel(obs)
    a = obs(i).pos;
    b = a + 1.5 * unit(obs(i).dir) * norm(a - q);
    plot3([a(1),b(1)]',[a(2),b(2)]',[a(3),b(3)]','b-');
  end
  axis equal;
end  

function [ a, b ] = min_segment( obs1, obs2 )
% MIN_SEGMENT returns the endpoints [a,b] of the shortest segment between
% two obsrvations obs1 and obs2 in 3D (with a on obs1 and b on obs2). 
% [ a, b ] = min_segment( obs1, obs2 )
  u = unit(obs1.dir); v = unit(obs2.dir);
  dot_uv = dot(u,v);
  A = [-1, dot_uv; dot_uv, -1];
  if rcond(A) <= eps
    a = p;
    b = obs2.pos - dot(obs2.pos - obs1.pos, u) * u;
  else
    b = [dot(u, obs1.pos - obs2.pos); ....
         dot(v, obs2.pos - obs1.pos)];
    t = A\b;
    a = obs1.pos + t(1) * u;
    b = obs2.pos + t(2) * v;
  end
end
function [] = test_min_segment()
  obs = struct('pos',{rand(3,1),rand(3,1)}, ...
               'dir',{rand(3,1),rand(3,1)});
  [a, b] = CityScan.min_segment(obs(1),obs(2));
  disp('These dot products should be close to zero');
  disp(['dot(u, a - b) = ', num2str(dot(obs(1).dir, a - b))]);
  disp(['dot(v, a - b) = ', num2str(dot(obs(2).dir, b - a))]);
  figure; hold on;
  d = norm(a - b);
  ps = [a + obs(1).dir * d, a, a - obs(1).dir * d];
  plot3(ps(1,:),ps(2,:),ps(3,:),'-or');
  ps = [b + obs(2).dir * d, b, b - obs(2).dir * d];
  plot3(ps(1,:),ps(2,:),ps(3,:),'-ob');
  ps = [a, b];
  plot3(ps(1,:),ps(2,:),ps(3,:),'-og');
  axis equal;
end


%-----------------------%
%   Ground Projection   %
%-----------------------%
% [ p, BW, H, R, T, P, lines ] = vanishing_point( I )
% [ vs ] = get_vanishing_points( filename, n, replace )
% [ I5, R5 ] = perspective_transform( I, f, s, p, t, cx, cy, fill )
% [ I3, A3 ] = place_image_on_image( I1, A1, I2, A2 )
% [ str ] = name_from_location( g, s, x, y )
% [ lib ] = lib_from_frame( I, f, s, p, ang, v, g)
% [] = import_lib_from_mov( filename, s, g, n )
% [] = plot_lib( filename, s, g )

function [ p, BW, H, R, T, P, lines ] = vanishing_point( I )
  gray = I;
  if numel(size(I)) == 3; gray = rgb2gray(I); end
  assert(numel(size(gray)) == 2, 'ERROR: input not a gray or RGB image');
  BW = edge(gray,'canny',[0.1,0.4],'both',1);
  [H,T,R] = hough(BW,'RhoResolution',0.5,'ThetaResolution',0.5);
  P  = houghpeaks(H,20,'threshold',ceil(0.2*max(H(:))));
  n = size(P,1);
  theta = T(P(:,2))';
  A = [cosd(theta),sind(theta)];
  b = R(P(:,1))';
  ps = cell(n);      % intersection points between lines
  for i = 1:n; for j = (i+1):n
    if rcond(A([i,j],:)) > eps
      ps{i,j} = A([i,j],:)\b([i,j]);
    end
  end; end
  t = max(size(I))/100;
  ds = struct([]);   % calculate distances between intersection points
  for i = 1:n; for j = (i+1):n
    if numel(ps{i,j}) == 2
      field = ['d' num2str(i) 'd' num2str(j)];
      for k = (i+1):n; for l = (k+1):n
        if (numel(ps{k,l}) == 2) && (norm(ps{i,j} - ps{k,l}) < t)
          if ~isfield(ds,field); ds(1).(field) = [i,j]; end
          ds(1).(field) = [ds(1).(field);[k,l]];
        end
      end; end
    end
  end; end
  fields = fieldnames(ds); m = 1;
  for i = 1:numel(fields)
    if numel(ds.(fields{i})) > numel(ds.(fields{m})); m = i; end
  end
  if m == 1
    disp('Error: No vanishing point found. Returning []');
    lines = []; p = []; return
  end
  lines = unique(ds.(fields{m}));
  p = A(lines,:)\b(lines);
end
function [] = test_vanishing_point()
  filename = 'ex/day_front/road1.mov';
  disp(['Using Example: ' filename]);
  CityScan.check_file_existence(filename,true);
  vid = VideoReader(filename);
  vid.CurrentTime = 240;
  I = flipud(permute(readFrame(vid),[2,1,3]));
  [p,BW,H,R,T,P,lines] = CityScan.vanishing_point(I);
  figure;
  set(gcf,'Color','white','Position',[100,100,1000,405])
  subplot('Position',[0,0,0.28,1]);
  imshow(-H'.^0.5,[],'XData',R,'YData',T);
  axis normal; hold on;
  x = R(P(:,1)); y = T(P(:,2)); 
  plot(x,y,'ob');
  plot(x(lines),y(lines),'og');
  xs = p(1)*cosd(T) + p(2)*sind(T);
  plot(xs,T,'-r');
  subplot('Position',[0.28,0,0.72,1]);
  im = imshow(I); hold on;
  set(im, 'AlphaData', BW*0.5+0.5)
  x = [1,size(I,2)]; l = 1;
  theta = T(P(:,2))';
  A = [cosd(theta),sind(theta)];
  b = R(P(:,1))';
  for i = 1:size(P,1)
    y = (b(i)-A(i,1).*x)./A(i,2);
    if i == lines(l)
      plot(x,y,'g-');
      l = min(numel(lines),l + 1);
    else
      plot(x,y,'b-');
    end
  end
  plot(p(1),p(2),'ro');
end

function [ v_points ] = get_vanishing_points( filename, n, replace )
  disp(['Using Example: ' filename]);
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  if ~replace && isfield(trial.mov,'v_points') && ... 
       numel(trial.mov.v_points) >= n
    v_points = trial.mov.v_points; return;
  end
  vid = VideoReader([filename '.mov']);
  i = 1;
  while hasFrame(vid)
    disp(['Calculating vanishing point #' num2str(i) '/' num2str(n)]);
    I = flipud(permute(readFrame(vid),[2,1,3]));
    v_points{i} = CityScan.vanishing_point(I);
    assert(numel(v_points{i}) == 2 || i ~= 1, ...
      'Error: cannot find a vanishing point for first frame');
    if i == n; break; end
    i = i + 1;
  end
  mov.v_points = v_points;
  disp(['Saving centroids to file: ' filename '.mat']);
  save([filename '.mat'],'ios','lp','mov');
end
function [] = test_get_vanishing_points()
  filename = 'ex/day_front/road1'; n = 10;
  vs = CityScan.get_vanishing_points(filename, n, false); 
end

function [ I5, R5 ] = perspective_transform( I, f, s, p, t, cx, cy, fill )
  if numel(fill) ~= 1
    I5 = [];
    for i = 1:numel(fill)
      [Ii,R5] = CityScan. ...
        perspective_transform(I(:,:,i),f,s,p,t,cx,cy,fill(i));
      I5 = cat(3,I5,Ii);
    end
    return;
  end
  x = p(1) * s; y = p(2) * s; z = p(3) * s;
  b = t(2); c = t(3); a = t(1) - pi / 2; 
  I1 = I(cy(1):cy(2),cx(1):cx(2));
  xlimits = cx - size(I,2)/2;
  ylimits = cy - size(I,1)/2;
  R1 = imref2d(size(I1),xlimits,ylimits);
  T = [cos(c), -sin(c), 0; sin(c), cos(c), 0; 0, 0, 1]';
  tform = affine2d(T);
  [I2,R2] = imwarp(I1,R1,tform,'FillValues',fill);
  T = [z,         0,          0; 
       0, -z*sin(b), z*f*cos(b); 
       0,    cos(b),   f*sin(b)]';
  tform = projective2d(T);
  [I3,R3] = imwarp(I2,R2,tform,'FillValues',fill);
  T = [cos(a), -sin(a), 0; sin(a), cos(a), 0; 0, 0, 1]';
  tform = affine2d(T);
  [I4,R4] = imwarp(I3,R3,tform,'FillValues',fill);
  T = [1, 0, x; 0, 1, y; 0, 0, 1]';
  tform = affine2d(T);
  [~,R_] = imwarp(I4,R4,tform);
  wx = round(R_.XWorldLimits);
  wy = round(R_.YWorldLimits);
  img_size = [wy(2) - wy(1),wx(2) - wx(1)];
  R_ = imref2d(img_size,wx,wy);
  [I5,R5] = imwarp(I4,R4,tform,'FillValues',fill,'OutputView',R_);
  R5.XWorldLimits = R5.XWorldLimits / s;
  R5.YWorldLimits = R5.YWorldLimits / s;
end
function [] = test_perspective_transform()
  filename = 'ex/day_front/road1';
  trial = CityScan.get_trial(filename);
  ios = trial.ios; t = trial.mov.times;
  vid = VideoReader([filename '.mov']);
  x = ios.f_lon_m(t(1:2)); y = ios.f_lat_m(t(1:2));
  ang = atan2(y(2)-y(1),x(2)-x(1)); 
  fig = figure;
  set(gcf,'Color','w','Position',[10,10,1400,600]);
  sp1 = subplot(1,2,1);    sp2 = subplot(1,2,2);
  set(sp1,'Position',[0,0,0.5,1]);
  set(sp2,'Position',[0.55,0.05,0.43,0.9]);
  I = flipud(permute(readFrame(vid),[2,1,3]));
  f = trial.mov.f; s = 100;
  disp('Calculating vanishing point from frame...');
  v  = CityScan.vanishing_point(I);
  cx = [1,size(I,2)];
  cy = [ceil(size(I,1)*0.5),size(I,1)*0.9];
  phi   = -atan((v(2) - size(I,1)/2)/f);
  as = [ang,phi,0]; p  = [0,0,0.9];
  disp('Calculating projection...');
  [I_,R_] = CityScan. ...
    perspective_transform(I, f, s, p, as, cx, cy,[0,0,0]);
  [A_, ~] = CityScan. ...
    perspective_transform(I(:,:,1)*0, f, s, p, as, cx, cy,255);
  A_ = uint8(A_ == 0) * 255;
  set(gcf,'CurrentAxes',sp1); hold on;
  imshow(I);                  plot([0,0;size(I,2),size(I,2)],[cy;cy],'b-');
  axis off; axis tight;
  set(gcf,'CurrentAxes',sp2); hold on;
  im = imshow(I_,R_);         plot(0,0,'ro','LineWidth',2,'MarkerSize',10);
  axis equal; axis tight;
  set(gca, 'YDir', 'normal','Color','none','FontSize',15);
  set(im, 'AlphaData',A_);
end

function [ I3, A3 ] = place_image_on_image( I1, A1, I2, A2 )
  for i = 1:max(1,size(I1,3))
    I3(:,:,i) = uint8(A1 ~= 0) .* I1(:,:,i) + ...
                uint8(A1 == 0) .* I2(:,:,i) .* uint8(A2 ~= 0);
  end
  A3 = max(A1,A2);
end
function [] = test_place_image_on_image()
  I(:,:,1) = uint8(checkerboard) * 255;
  I(:,:,2) = 255 - I(:,:,1);
  A = I * 0;
  A(1:55,1:55,1) = 255;
  A(25:end,25:end,2) = 255;
  [I(:,:,3),A(:,:,3)] = CityScan. ...
    place_image_on_image(I(:,:,1),A(:,:,1),I(:,:,2),A(:,:,2));
  figure; set(gcf,'Color','c');
  for i = 1:3
    subplot(1,3,i);
    im = imshow(I(:,:,i)); axis on;
    set(im,'AlphaData',A(:,:,i));
    set(gca,'Color','none');
  end
end

function [ str ] = name_from_location( g, s, x, y )
  assert(g - round(g) == 0,'Error: First argument must be an integer');
  x_ = floor(x / g);
  y_ = floor(y / g);
  str = ['g' num2str(g) 's' num2str(s) '|' ...
         sprintf('%+d',x_) '|' sprintf('%+d',y_)];
end
function [] = test_name_from_location()
  disp(CityScan.name_from_location(20,100,1348134.32,571987.94))
end

function [ lib ] = lib_from_frame( I, f, s, p, ang, v, g )
  cx = [1,size(I,2)];
  cy = [ceil(size(I,1)*0.5),size(I,1)*0.9];
  phi   = -atan((v(2) - size(I,1)/2)/f);
  as = [ ang, phi,  0];
  disp('Calculating projection...');
  [I_,R_] = CityScan.perspective_transform(I, f, s, p, as, cx, cy,[0,0,0]);
  A_ = CityScan.perspective_transform(I(:,:,1)*0, f, s, p, as, cx, cy,255);
  disp('Projection done...');
  A_ = uint8(A_ == 0) * 255;
  xl = R_.XWorldLimits;             yl = R_.YWorldLimits;
  cx = floor( xl(1) /  g) * g;      cy = floor( yl(1) /  g) * g;
  w  =  ceil((xl(2) - cx) / g);     h  =  ceil((yl(2) - cy) / g);
  x  = round((xl(1) - cx) * s) + 1; y  = round((yl(1) - cy) * s) + 1;
  A  = uint8(zeros([h,w] * g * s)); B(:,:,3) = A;
  xr = x:(x+size(I_,2)-1);          yr = y:(y+size(I_,1)-1);
  A(yr,xr)   = A_;                  B(yr,xr,:) = I_;
  lib = struct; k = 1;
  for i = 1:w; for j = 1:h
    rx = ((i - 1) * g * s + 1):(i * g * s);
    ry = ((j - 1) * g * s + 1):(j * g * s);
    if any(any(A(ry,rx) ~= 0))
      lib(k).x = cx / g + i - 1; lib(k).I = B(ry,rx,:); 
      lib(k).y = cy / g + j - 1; lib(k).A = A(ry,rx);
      lib(k).name = CityScan. ...
        name_from_location(g,s,lib(k).x * g, lib(k).y * g);
      lib(k).R = imref2d(size(lib(k).A),[lib(k).x,lib(k).x + 1] * g, ...
                                        [lib(k).y,lib(k).y + 1] * g);
      k = k + 1;
    end
  end; end
end
function [] = test_lib_from_frame()
  filename = 'ex/day_front/road1';
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  t = mov.times;
  vid = VideoReader([filename '.mov']);
  f = trial.mov.f; s = 100; g = 10;
  x = ios.f_lon_m(t(1:2));
  y = ios.f_lat_m(t(1:2));
  ang = atan2(y(2)-y(1),x(2)-x(1)); 
  I = flipud(permute(readFrame(vid),[2,1,3]));
  p = [x(1),y(1),0.9];
  disp('Calculating vanishing point from frame...');
  v = CityScan.vanishing_point(I);
  lib = CityScan.lib_from_frame(I,f,s,p,ang,v,g);
  for frame = lib
    hold on;
    im = imshow(frame.I,frame.R);
    set(im,'AlphaData',max(frame.A,50));
  end
  set(gca,'YDir','normal');
end

function [] = import_lib_from_mov( filename, s, g, n )
  tic;
  trial = CityScan.get_trial(filename);
  v_points = CityScan.get_vanishing_points(filename,n,false); 
  trial = CityScan.get_trial(filename);
  ios = trial.ios; lp = trial.lp; mov = trial.mov;
  t = mov.times; f = mov.f;
  disp(['Saving trial data to file: ' filename '.mat']);
  save([filename '.mat'],'ios','lp','mov');
  vid = VideoReader([filename '.mov']);
  x = ios.f_lon_m(t); y = ios.f_lat_m(t);
  ang = CityScan.ang_from_xy(x,y);
  lib = struct('name',{});
  for fi = 1:n
    delete = ones(size(lib));
    if hasFrame(vid)
      disp(['Processing frame #' num2str(fi) '/' num2str(n) '...']);
      I = flipud(permute(readFrame(vid),[2,1,3]));
      p = [x(fi),y(fi),0.9]; 
      if numel(v_points{fi}) == 2; v = v_points{fi}; end
      l = CityScan.lib_from_frame(I,f,s,p,ang(fi),v,g);
      if fi == 1; lib = l; continue; end
      temp = lib(1); k = 1; 
      for i = 1:numel(l)
        add = true;
        for j = 1:numel(lib)
          if (l(i).x == lib(j).x) && (l(i).y == lib(j).y)
            % Add imread to process paths returning to a grid pixel
            temp(k) = lib(j);
            [temp(k).I,temp(k).A] = CityScan. ...
              place_image_on_image(l(i).I,l(i).A,lib(j).I,lib(j).A);
            if fi ~= n; delete(j) = 0; end
            add = false; k = k + 1; break;
          end
        end
        if add; temp(k) = l(i); k = k + 1; end
      end
    end
    for j = 1:numel(lib); if delete(j) == 1
      file = [filename '/' lib(j).name '.png'];
      disp(['Writing image: ' file]);
      if exist(filename) == 0; mkdir(filename); end
      imwrite(lib(j).I,file,'Alpha',lib(j).A);
    end; end
    if ~hasFrame(vid); break; end
    lib = temp;
  end; toc
end
function [] = test_import_lib_from_mov()
  filename = 'ex/day_front/road1';
  s = 20; g = 10; n = 10;
  CityScan.import_lib_from_mov(filename,s,g,n);
end

function [ bounds ] = get_street_typology( filename, s, g, x, y, ang )
  lib = dir([filename '/']); r = 2;
  files = {};
  for i = 1:numel(lib)
    header = ['g' num2str(g) 's' num2str(s) '|'];
    file = lib(i).name;
    if (numel(file) > numel(header)) && strcmp(header,file(1:numel(header)))
      coords = strsplit(file((numel(header)+1):end),{'|','.'});
      cx = str2num(coords{1});
      cy = str2num(coords{2});
      if ((cx > x - r) && (cx + 1 < x + r) && ...
          (cy > y - r) && (cy + 1 < y + r))
        files = [files {{cx,cy,[filename '/' file]}}];
      end
    end
  end
  cxmin = floor(x - 1);
  cymin = floor(y - 1);
  I_full = uint8(zeros(3 * s * g,3 * s * g,3));
  for i = 1:numel(files)
    sx = (files{i}{1} - cxmin) * s * g + 1;
    sy = (files{i}{2} - cymin) * s * g + 1;
    [I,~,A] = imread(files{i}{3});
    I_full(sy:(sy + g * s - 1),sx:(sx + g * s - 1),:) = I;
  end
  xc = (x - cxmin) * g * s;
  yc = (y - cymin) * g * s;
  d = g * s;
  s1 = subplot(2,2,1);
  set(s1,'Position',[0.05,0.55,0.4,0.4]);
  imshow(I_full); hold on;
  plot(g * s * ([0;1;2;3] * ones(1,2))', ...
       g * s * [0;3] * ones(1,4),'w-','LineWidth',2);
  plot(g * s * [0;3] * ones(1,4), ...
       g * s * ([0;1;2;3] * ones(1,2))','w-','LineWidth',2);
  plot([xc - d,xc + d,xc + d,xc - d,xc - d], ...
       [yc - d,yc - d,yc + d,yc + d,yc - d],'r-','LineWidth',2);
  plot(xc,yc,'ro');
  I_full = I_full(round(yc - d):round(yc + d), ...
                  round(xc - d):round(xc + d),:);
  px = [-1,-1,1,1,-1] * g * s / sqrt(2);
  py = [-1,1,1,-1,-1] * g * s / sqrt(2);
  ang = ang + pi / 2;
  rx = px * cos(ang) - py * sin(ang) + g * s;
  ry = py * cos(ang) + px * sin(ang) + g * s;
  s2 = subplot(2,2,2);
  set(s2,'Position',[0.55,0.55,0.4,0.4]);
  imshow(I_full); hold on;
  plot(rx,ry,'r-','LineWidth',2);
  plot(g * s,g * s,'ro');
  T = [cos(ang), -sin(ang), 0; sin(ang), cos(ang), 0; 0, 0, 1];
  tform = affine2d(T);
  [I_rot] = imwarp(I_full,tform);
  I_rot = im2bw(I_rot,0.8);
  [h,w] = size(I_rot);
  off = g * s / sqrt(2);
  I_rot = I_rot(round(h / 2 - off):round(h / 2 + off), ...
                round(w / 2 - off):round(w / 2 + off));
  s3 = subplot(2,2,3);
  set(s3,'Position',[0.05,0.05,0.4,0.4]);
  imshow(I_rot);
  signal = sum(I_rot) > 350;
  lines = [];
  j = 0;
  for i = 1:numel(signal)
    if signal(i) == 0
      if j ~= 0
        w = i - j - 1;
        lines = [lines round(j + w / 2)];
        j = 0; 
      end
    end
    if signal(i) ~= 0 && j == 0; j = i; end
  end
  bounds = [];
  for i = 2:numel(lines)
    width = lines(i) - lines(i - 1);
    if width > 200 && width < 300
      bounds = [bounds, lines(i), lines(i - 1)];
    end
  end
  s4 = subplot(2,2,4);
  set(s4,'Position',[0.55,0.05,0.4,0.4]);
  plot(signal,'b-');
  hold on;
  for i = 1:numel(bounds)
    plot([bounds(i),bounds(i)],[0,1],'r-','Linewidth',2);
  end
  axis tight;
  axis square;
  axis off;
  str = '|';
  for i = 2:2:numel(bounds)
    str = [str '| ' num2str((bounds(i - 1) - bounds(i)) / 100) ' m |'];
  end
  str = [str '|'];
  title(str,'FontSize',15);
end
function [] = test_get_street_typology()
  filename = 'ex/day_front/road2';
  s = 100; g = 10; fps = 30;
  trial = CityScan.get_trial(filename);
  ts = 1:(1 / fps):max(trial.ios.time_s);
  xt = trial.ios.f_lon_m(ts); yt = trial.ios.f_lat_m(ts);
  at = CityScan.ang_from_xy(xt,yt);
  % CityScan.import_lib_from_mov(filename,s,g,n);
  vid = VideoWriter('street_typology','MPEG-4');
  vid.FrameRate = fps;
  open(vid); 
  fig = figure;
  set(fig,'Color','w','Position',[20,20,800,800]); 
  for i = (221 * fps):(235 * fps)
    clf;
    x = xt(i) / g; y = yt(i) / g; ang = at(i);
    CityScan.get_street_typology(filename,s,g,x,y,ang);
    drawnow;
    img = frame2im(getframe(fig));
    writeVideo(vid,img);
  end
  close(vid);
end


function [] = plot_lib( filename, s, g, x, y, r )
  figure;
  lib = dir([filename '/']);
  for i = 1:numel(lib)
    header = ['g' num2str(g) 's' num2str(s) '|'];
    file = lib(i).name;
    if (numel(file) > numel(header)) && strcmp(header,file(1:numel(header)))
      coords = strsplit(file((numel(header)+1):end),{'|','.'});
      cx = str2num(coords{1});
      cy = str2num(coords{2});
      if (r == 0) || ((cx > x - r) && (cx + 1 <= x + r) && ...
                      (cy > y - r) && (cx + 1 <= x + r))
        [I,~,A] = imread([filename '/' file]);
        R  = imref2d(size(A),[cx,cx + 1] * g,[cy,cy + 1] * g);
        im = imshow(I,R); hold on;
        set(im,'AlphaData',max(A,50));
      end
    end
  end
  axis on; grid on; axis tight;
  set(gcf,'Position',[200,200,1000,800]);
  set(gca,'YDir','normal','Color','none','GridColor','r', ...
          'Layer','top','OuterPosition',[0 0 1 1]);
  xl = get(gca,'XLim');
  yl = get(gca,'YLim');
  set(gca,'XTick',xl(1):g:xl(2),'YTick',yl(1):g:yl(2));
  set(gca,'XTickLabelRotation',90);
end
function [] = test_plot_lib()
  filename = 'ex/day_front/road2';
  s = 100; g = 10; x = -12797; y = 421809; r = 4;
  % CityScan.import_lib_from_mov(filename,s,g,n);
  CityScan.plot_lib(filename,s,g,x,y,r);
end


end
end

function [ u ] = unit( v )
  assert(norm(v) ~= 0,'ERROR: Dividing by zero');
  u = v/norm(v);
end
