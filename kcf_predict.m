function [ result,tracker ] = kcf_predict( I, tracker)
%I is the gray image and bb = [x1,y1,x2,y2]
peak_values = zeros(3,1);
scales = zeros(3,1);
results_delta = zeros(3,2);

target_sz = [tracker.bb(4)-tracker.bb(2),tracker.bb(3)-tracker.bb(1)];
pos = [tracker.bb(2), tracker.bb(1)] + target_sz/2;

cont = 1;
for scale_adjust = 0.95:0.05:1.05
%for scale_adjust = 1
    window_sz = floor(tracker.window_sz*tracker.scale*scale_adjust);
    %obtain a subwindow for detection at the position from last
    %frame, and convert to Fourier domain (its size is unchanged)
    patch = get_subwindow(I, pos, window_sz);
    if(size(patch,1)~=tracker.window_sz(1)||size(patch,2)~=tracker.window_sz(2))
        patch = imResample(patch, tracker.window_sz, 'bilinear');
    end
    zf = fft2(get_features(patch, tracker.features, tracker.cell_size, tracker.cos_window));
    kzf = gaussian_correlation(zf, tracker.model_xf, tracker.kernel.sigma);
    response = real(ifft2(tracker.model_alphaf .* kzf));  %equation for fast detection
    
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - size(zf,1);
    end
    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
        horiz_delta = horiz_delta - size(zf,2);
    end
    
    peak_value = max(response(:));
    if(scale_adjust~=1)
        peak_value = peak_value * 0.95;
    end
    peak_values(cont)=peak_value;
    scales(cont)= tracker.scale*scale_adjust;
    results_delta(cont,:) = [vert_delta-1,horiz_delta-1];
    cont = cont+1;
end
index = find(peak_values == max(peak_values));
tracker.scale = scales(index);
pos = pos + tracker.cell_size * results_delta(index,:); 
target_sz = tracker.init_target_sz * tracker.scale;
x1 = pos(2)-target_sz(2)/2;
y1 = pos(1)-target_sz(1)/2;
x2 = pos(2)+target_sz(2)/2;
y2 = pos(1)+target_sz(1)/2;

result = [x1,y1,x2,y2];

%fprintf('predict scale is %d\n',tracker.scale);
end

