function indices = hidden_indices(directory)
	indices = [];
	for i=1:length(directory)
		if directory(i).name(1) == '.'
			indices(end+1) = i;
		end
	end
end