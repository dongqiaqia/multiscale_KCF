function [tracker ] = kcf_initialize( I, bb,tracker,opt)
%kcf initialize,the input is gray image,target box and coefficency.

cell_size = opt.cell_size;
features = opt.features;
kernel = opt.kernel;
padding = opt.padding;
output_sigma_factor = opt.output_sigma_factor;
lambda = opt.lambda;
template_sz = opt.template_sz;
scale = 1;

%set initial position and size
target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
pos = [bb(2), bb(1)] + target_sz/2;


    
%if the target is large, lower the resolution, we don't need that much
%detail
if (sqrt(prod(target_sz)) >= template_sz)
    if(target_sz(1)>target_sz(2))
        scale = target_sz(1)/template_sz;
    else
        scale = target_sz(2)/template_sz;
    end
end
target_sz = floor(target_sz/scale);

%window size, taking padding into account
window_sz = floor(target_sz * (1 + padding));

%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

%store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

if size(I,3) > 1,
    I = rgb2gray(I);
end

extracted_sz = window_sz * scale;

%obtain a subwindow for training at newly estimated target position
patch = get_subwindow(I, pos, extracted_sz);
if(size(patch,1)~=window_sz(1)||size(patch,2)~=window_sz(2))
    patch = imResample(patch, window_sz, 'bilinear');
end
xf = fft2(get_features(patch, features, cell_size, cos_window));

kf = gaussian_correlation(xf, xf, kernel.sigma);
alphaf = yf ./ (kf + lambda);   %equation for fast training
model_alphaf = alphaf;
model_xf = xf;

tracker.cell_size = opt.cell_size;
tracker.features = opt.features;
tracker.kernel = opt.kernel;
tracker.padding = opt.padding;
tracker.output_sigma_factor = opt.output_sigma_factor;
tracker.interp_factor = opt.interp_factor;
tracker.lambda = opt.lambda;

tracker.model_alphaf = model_alphaf;
tracker.model_xf = model_xf;
tracker.cos_window = cos_window;
tracker.window_sz = window_sz;
tracker.yf = yf;
tracker.scale = scale;
tracker.bb = bb;
tracker.init_target_sz = target_sz;
%fprintf('init scale is %d\n',tracker.scale);

end

