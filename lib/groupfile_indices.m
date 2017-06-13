function [indices] = groupfile_indices(directory, string)
    
    indices = [];
    for i=1:length(directory)
        if strfind(directory(i).name, string)
            indices(end+1) = i;
        end
    end
end
