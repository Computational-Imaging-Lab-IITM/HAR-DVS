function generate_motion_maps(dataset_path, features_path)
    %% load the data directory %%
    data_dir = dir(dataset_path);

    %% remove the hidden files from it %%
    inds = hidden_indices(data_dir);
    data_dir(inds) = [];

    %% read video and aveage along the three axes %%
    tic
    for class=1:length(data_dir)
        class_dir_videos = dir(fullfile(dataset_path, data_dir(class).name, '/*.avi'));
        num_videos = length(class_dir_videos);

        fprintf('Generating motion maps for class = %d\n', class);
        parfor i=1:num_videos
            filename = fullfile(dataset_path, data_dir(class).name, class_dir_videos(i).name);
            video = VideoReader(filename);
    %         fps = video.FrameRate;
    %         duration = video.Duration;
            time = 0;

            videoFrames = [];
            xy_videoFrames = [];
            xt_videoFrames = [];
            yt_videoFrames = [];

    %         parfor time=1:round((duration*fps))-1
    %             video = VideoReader(filename);
    %             fps = video.FrameRate;
    %             duration = video.Duration;
    % 
    %             video.CurrentTime = time/fps;    
    % 
    %             if hasFrame(video)
    %                 frame = rgb2gray(readFrame(video));
    %                 videoFrames(time, :, :) = frame(:, :);
    %             end
    %         end

            while hasFrame(video)
                time = time +1;
                temp = rgb2gray(readFrame(video));
                videoFrames(time, :, :) = temp(:, :);
            end

            xy_videoFrames(:, :) = sum(videoFrames, 1);
            xt_videoFrames(:, :) = sum(videoFrames, 2);
            yt_videoFrames(:, :) = sum(videoFrames, 3);
            
            output_filename = fullfile(features_path, data_dir(class).name, class_dir_videos(i).name);
            
            filename_xy = regexprep(output_filename, '.avi', '_xy.jpg');
            filename_xt = regexprep(output_filename, '.avi', '_xt.jpg');
            filename_yt = regexprep(output_filename, '.avi', '_yt.jpg');

            disp(class_dir_videos(i).name);
            imwrite(normalize_image(xy_videoFrames), filename_xy, 'JPEG');
            imwrite(normalize_image(xt_videoFrames), filename_xt, 'JPEG');
            imwrite(normalize_image(yt_videoFrames), filename_yt, 'JPEG');
        end
        fprintf('\n');
    end
    toc
end