function [positions, time] = multiscale_tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%scaled kcf tracker

opt.cell_size = cell_size;
opt.features = features;
opt.kernel = kernel;
opt.padding = padding;
opt.output_sigma_factor = output_sigma_factor;
opt.interp_factor = interp_factor;
opt.lambda = lambda;
opt.template_sz = 96;

x1 = pos(2)-target_sz(2)/2;
y1 = pos(1)-target_sz(1)/2;
x2 = pos(2)+target_sz(2)/2;
y2 = pos(1)+target_sz(1)/2;

bb = [x1,y1,x2,y2];
tracker = [];

if show_visualization,  %create video interface
    update_visualization = show_video(img_files, video_path,0);
end


time = 0;  %to calculate FPS
positions = zeros(numel(img_files), 2);  %to calculate precision

for frame = 1:numel(img_files)

    
    im = imread([video_path img_files{frame}]);
    if size(im,3) > 1,
        im = rgb2gray(im);
    end
    
    tic()
   if frame == 1
       tracker = kcf_initialize(im,bb,tracker,opt);
   else
       [bb,tracker] = kcf_predict(im,tracker);
       tracker.bb = bb;
       tracker = kcf_update(im,tracker);
   end
   target_sz = [bb(4)-bb(2),bb(3)-bb(1)];
   pos = [bb(2), bb(1)] + floor(target_sz/2);		%save position and timing
    positions(frame,:) = pos;
    time = time + toc();
   
    %visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early

        drawnow
    % 			pause(0.05)  %uncomment to run slower
    end
    
end

end



