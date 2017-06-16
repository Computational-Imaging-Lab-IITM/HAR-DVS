function [indices] = groupfile_indices_mm(files, string)
    
    indices = [];
    for i=1:length(files)
        if strfind(files{i}, string)
            indices(end+1) = i;
        end
    end
end
