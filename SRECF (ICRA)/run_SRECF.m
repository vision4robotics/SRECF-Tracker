

function results = run_SRECF(seq)

%   HOG feature parameters
hog_params.nDim   = 31;
hog_params.cell_size = 4;
%   ColorName feature parameters
cn_params.nDim  =10;
cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
%  Grayscale feature parameters
grayscale_params.nDim=1;
grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

%   Global feature parameters 
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
  };
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       =0.025;        % learning rate  0.013
params.output_sigma_factor = 0.064;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 50;           % number of Newton's iteration to maximize the detection scores
				% the weight of the standard (uniform) regularization, only used when params.use_reg_window == 0
%   Scale parameters
params.number_of_scales = 33 ;
params.scale_sigma_factor=0.475;
params.hog_scale_cell_size = 4;  
params.scale_step=1.03;
params.scale_model_factor = 1.0;
params.learning_rate_scale=0.025;
params.scale_model_max_area = 32*16;

params.scale_lambda = 1e-4;      


%occlusion params
params.tau=0.065;
params.M_learning_thre=0.04;
params.L=7.2;
params.repression_limit=0.11;


%   size, position, frames initialization
params.video_path = seq.video_path;
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram = numel(params.s_frames);
params.s_name=seq.name;
%   ADMM parameters, # of iteration, and lambda- mu and betha are set in

%   the main function.
params.admm_iterations = 5;
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;

%   Run the main function
results = tracking(params);
