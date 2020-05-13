function [results] = tracking(params)

%   Setting parameters for local use.
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
num_scales          = params.number_of_scales;
scale_sigma_factor  =params.scale_sigma_factor;
scale_step          = params.scale_step;
scale_lambda        =params.scale_lambda;   
scale_model_factor  =params.scale_model_factor;
scale_model_max_area =params.scale_model_max_area;
interpolate_response = params.interpolate_response;
newton_iterations = params.newton_iterations;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

%params for score pool
tau=params.tau;
M_learning_thre=params.M_learning_thre;
L=params.L;
repression_limit=params.repression_limit;


%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;                 %4
search_area = prod(init_target_sz / featureRatio * search_area_scale);      %将初始目标大小除以提取特征时的cells大小，再乘以搜索域比，获得搜索区域的cells数目

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area         %如果当前搜索区域cells数比cells阈值乘以最大允许搜索域cells的乘积还小，说明此时cells数目不够多 
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;       %全局特征参数：cells尺寸，以及最小特征阈值

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio，按2500最大搜索域缩放过的目标尺寸，所对应的窗口尺寸
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

width=size(y);
width=width(1);
centre_crop=ones(width);
centre_crop(1,1)=0;
centre_crop=fftshift(centre_crop(:,:));
[id_ymax_peak, id_xmax_peak] = find(centre_crop == min(centre_crop(:)));
centre_crop(id_ymax_peak-2:id_ymax_peak+2,id_xmax_peak-2:id_xmax_peak+2)=0;
[T,W]=find(y==max(max(y)));
y_model=circshift(y,[id_ymax_peak,id_xmax_peak]-[T(1),W(1)]);
% surf(y_model);

M_set=cell(10,1);
M_max=cell(10,1);
for i=1:10
    M_set{i}=zeros(size(y));
    M_max{i}=0;
end


% M1=zeros(size(y));
% M2=zeros(size(y));
% M3=zeros(size(y));
% M4=zeros(size(y));
% M5=zeros(size(y));
% M6=zeros(size(y));
% M7=zeros(size(y));
% M8=zeros(size(y));
% M9=zeros(size(y));
% M10=zeros(size(y));

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% Calculate feature dimension
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});     %像素值的长宽，以及RGB三通道
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end


%% SCALE ADAPTATION INITIALIZATION
% Use the translation filter to estimate the scale
scale_sigma = sqrt(num_scales) * scale_sigma_factor;
ss = (1:num_scales) - ceil(num_scales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));     %desired label for scale estimation 
if mod(num_scales,2) == 0
    scale_window = single(hann(num_scales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(num_scales));
end
ss = 1:num_scales;
scaleFactors = scale_step.^(ceil(num_scales/2) - ss);
if scale_model_factor^2 * prod(target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));
end
if prod(target_sz) >scale_model_max_area
    params.scale_model_factor = sqrt(scale_model_max_area/prod(target_sz));
end
scale_model_sz = floor(target_sz * scale_model_factor);

% set maximum and minimum scales
min_scale_factor = scale_step ^ ceil(log(max(5 ./sz)) / log(scale_step));
max_scale_factor =scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
    
% Pre-computes the grid that is used for score optimization
ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';

% initialize the projection matrix (x,y,h,w)
rect_position = zeros(num_frames, 4);
time = 0;
loop_frame = 1;

small_filter_sz = floor(base_target_sz/featureRatio);

for frame = 1:numel(s_frames)%开始第一帧
    %load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    tic();
    
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    if frame > 1
          pixel_template=get_pixels(im, pos, round(sz*currentScaleFactor), sz);            
          xt=get_features(pixel_template,features,global_feat_params);
          xtf=fft2(bsxfun(@times,xt,cos_window));        
          responsef=permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
          % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        responsef_padded = resizeDFT2(responsef, interp_sz);      
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        % find maximum peak
        [disp_row, disp_col] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        
        % calculate translation
        translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
        
        % update position
        old_pos = pos;
        pos = pos + translation_vec;
        
        [t_m,w_m]=find(response==max(max(response)));
        Mc_max=response(t_m,w_m);
        Mc=circshift(response,[id_ymax_peak,id_xmax_peak]-[t_m(1),w_m(1)]);
        %%Scale Search
            xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz,features(2),global_feat_params);
            xsf = fft(xs,[],2);
            scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den+scale_lambda)));            
            % find the maximum scale response
            recovered_scale = find(scale_response == max(scale_response(:)), 1);
            % update the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        %score pool is perfomed belowed
        [t_m,w_m]=find(response==max(max(response)));
        Mc_max=response(t_m,w_m);
        Mc=circshift(response,[id_ymax_peak,id_xmax_peak]-[t_m(1),w_m(1)]);
        if frame>10
            M_all=zeros(size(y));
            M_max_all=0;
            delta=zeros(size(y));
            for i=1:10
                M_all=M_set{i}+M_all;
                M_max_all=M_max{i}+M_max_all;
            end
            M_ave=M_all/10;
%             figure(4);
%             surf(M_ave);
%             figure(3);
%             surf(Mc);
            M_max_ave=M_max_all/10;
            delta=(Mc-M_ave)/M_max_ave;%the robustness of response
%             figure(3);
%             surf(delta);
            mask_gamma_ne=ones(width);
            mask_gamma_ne(Mc<0)=0;
            gamma=Mc-M_ave;
            gamma(delta<tau)=0;
            gamma_mask=imregionalmax(gamma);
            gamma=gamma.*gamma_mask.*centre_crop.*mask_gamma_ne;
%             gamma=gamma.*centre_crop.*mask_gamma_ne;
            Mc_max=Mc_max(1,1);
            alpha=L.*exp(1./sqrt(1-(Mc/Mc_max).*centre_crop.*(Mc/Mc_max))).*(1./(1-delta)).^2;%the degree of repression
            alpha(alpha>50)=50;
            alpha(alpha==inf)=50;
            num_noise=sum(sum(gamma~=0));
            [m,n]=size(gamma);
            v=sort(reshape(gamma,1,m*n),'descend');
            if num_noise>=22
                v=v(1:22);
            else
                v=v(1:num_noise);
            end
            idx=[];
            for i=1:length(v)
                idx=[idx; find(gamma==v(i))];
            end
            kk=zeros(size(gamma));
            kk(idx)=1;
            gamma=gamma.*kk;
            repression=gamma.*alpha;
%             repression(repression>repression_limit)=repression_limit;
            repression(repression>repression_limit)=repression_limit;
%             figure(6);
%             surf(repression);
            repression_shift=circshift(repression,-([id_ymax_peak,id_xmax_peak]-[1,1]));
            y_train=y-repression_shift;
            y_train_f= fft2(y_train); %   FFT of y.
        end

    end

%     frame
if frame==1   
    % extract training sample image region
        pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
        pixels = uint8(gather(pixels));
        x=get_features(pixels,features,global_feat_params);
        xf=fft2(bsxfun(@times,x,cos_window));
        xcn_template=x(:,:,32:41);
 else
    % use detection features
         shift_samp_pos = 2*pi * translation_vec ./(currentScaleFactor* sz);
         xf = shift_sample(xtf, shift_samp_pos, kx', ky');
end
%%
    %此处加入用颜色特征判断是否产生遮挡（，则不更新模板）
    %如果未遮挡，则按算法分配不同区域的学习率2019.11.22
    
%     if frame > 1
%     cnpixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
%     cnpixels = uint8(gather(cnpixels));
%     xcnt=get_features(cnpixels,features(3),global_feat_params);
% %     xcnt=get_features(cnpixels,features,global_feat_params);
%     
% 
%     L=use_sz(1);
%     P=zeros(L);
% 
%     interest_region_size=ceil(base_target_sz./4);
%     
%     a1=floor((L-interest_region_size(2))/2);
%     a2=ceil(L/2+interest_region_size(2)/2);
%     b1=floor((L-interest_region_size(1))/2);
%     b2=ceil(L/2+interest_region_size(1)/2);
%     P(b1:b2,a1:a2)=1;
% 
%     end
%     if occ==true
        
%     do nothing 
%     else
    if frame <=10
        y_train_f=yf;
    end

    if (frame == 1)
        model_xf = xf;
    else
        
        model_xf = ((1 - learning_rate) * model_xf) + (learning_rate * xf);
    end
    
    g_f = single(zeros(size(xf)));
    h_f = g_f;
    l_f = g_f;
    mu    = 1;
    betha = 10;
    mumax = 10000;
    i = 1;
    
    T = prod(use_sz);
    S_xx = sum(conj(model_xf) .* model_xf, 3);
    params.admm_iterations = 2;
    %   ADMM
    while (i <= params.admm_iterations)
        %   solve for G- please refer to the paper for more details
        B = S_xx + (T * mu);
        S_lx = sum(conj(model_xf) .* l_f, 3);
        S_hx = sum(conj(model_xf) .* h_f, 3);
        g_f = (((1/(T*mu)) * bsxfun(@times, y_train_f, model_xf)) - ((1/mu) * l_f) + h_f) - ...
            bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf, (S_xx .* y_train_f))) - ((1/mu) * bsxfun(@times, model_xf, S_lx)) + (bsxfun(@times, model_xf, S_hx))), B);
        
        %   solve for H
        h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
        [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
        t(sx,sy,:) = h;
        h_f = fft2(t);
        
        %   update L
        l_f = l_f + (mu * (g_f - h_f));
        
        %   update mu- betha = 10.
        mu = min(betha * mu, mumax);
        i = i+1;
    end
    
    if frame<=10
    responsef=permute(sum(bsxfun(@times, conj(g_f), model_xf), 3), [1 2 4 3]);
    % if we undersampled features, we want to interpolate the
    % response so it has the same size as the image patch
    responsef_padded = resizeDFT2(responsef, interp_sz);      
    % response in the spatial domain
    response = ifft2(responsef_padded, 'symmetric');
    [t_m,w_m]=find(response==max(max(response)));
    Mc_max=response(t_m,w_m);
    Mc=circshift(response,[id_ymax_peak,id_xmax_peak]-[t_m(1),w_m(1)]);
    M_set{frame}=Mc;
    M_max{frame}=Mc_max;
    end
    if frame > 10
        set_update_mask=ones(size(y));
        set_update_mask(delta>M_learning_thre)=0;
        for i=1:10
            if i<10
               M_set{i}=M_set{i}.* (1-set_update_mask)+set_update_mask.*M_set{i+1};
               M_max{i}=M_max{i+1};
            elseif i==10
                M_set{10}=M_set{10}.*(1-set_update_mask)+set_update_mask.*Mc;
                M_max{10}=Mc_max;
            end
        end  
    end
    
    
    
    %% Upadate Scale
    if frame==1
        xs = crop_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz,features(2),global_feat_params);
    else
        xs= shift_sample_scale(im, pos, base_target_sz,xs,recovered_scale,currentScaleFactor*scaleFactors,scale_window,scale_model_sz,features(2),global_feat_params);
    end
    
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - params.learning_rate_scale) * sf_den + params.learning_rate_scale * new_sf_den;
        sf_num = (1 - params.learning_rate_scale) * sf_num + params.learning_rate_scale * new_sf_num;
    end
    
%     end
    % Update the target size (only used for computing output box)
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position and calculate FPS
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc();
    
%%
%  visualization
     if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        figure(1);
        imshow(im);
        if frame == 1
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 26, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
        else
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(12, 28, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            text(12, 66, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 12);
            hold off;
         end
        drawnow
    end
     loop_frame = loop_frame + 1;
    %visualization
%     if visualization == 1
%         rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%         im_to_show = double(im)/255;
%         if size(im_to_show,3) == 1
%             im_to_show = repmat(im_to_show, [1 1 3]);
%         end
%         if frame == 1
%             fig_handle = figure('Name', 'Tracking');
%             imagesc(im_to_show);
%             hold on;
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(frame), 'color', [0 1 1]);
%             hold off;
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%         else
%             resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
%             sc_ind = floor((nScales - 1)/2) + 1;
%             
%             figure(fig_handle);
%             imagesc(im_to_show);
%             hold on;
%             resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
%             alpha(resp_handle, 0.2);
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
%             text(20, 60, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
%             
%             hold off;
%         end
%         drawnow
%     end
%     loop_frame = loop_frame + 1;
end
%   save resutls.
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
