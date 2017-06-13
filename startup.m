%% clear workspace %%
clear;
close;
clc;

%% get the current folder and cd to it %%
currentFolder = regexprep(mfilename('fullpath'), mfilename(), '');
cd(currentFolder);

%% add the current folder and subfolders to path %%
addpath(genpath(currentFolder));
