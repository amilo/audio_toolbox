
inputdir = '/Users/amilo/Desktop/karl/P01/yourfolder/filt/';
S = dir(fullfile(inputdir,'*.wav'));

Fs=44100;  
[HdAw, HdBw, HdCw] = ABCdsgn(Fs);

for k = 1:numel(S)
    fnm = fullfile(inputdir,S(k).name);
    disp(fnm);    
    [x,Fs] = audioread(fnm);
    y = filter(HdAw, x);
    filename = strcat(fnm(1:end-4), '_A.wav');
    disp(strcat('filtered file: ', filename));
    audiowrite(filename,y,Fs);
    ...
end








% 
% for k = 1:numel(S)
%     fnm = fullfile(inputdir,S(k).name);
% %     txt = load(fnm);
%     filter(fnm)
%     disp(fnm);
%     ...
% end
% file='/Users/amilo/Desktop/karl/participants/yourfolder/251119_P1-left.wav';
%  filter(file);
%  
%  function filter = filterA(file)
 

% function filter(file)



% HdAw

% B = coeffs(HdAw)

% B.Stage1.Stage1
% B.Stage1.Stage2
% B.Stage2


% Create a time record
% buf=randn(1, 44100);
% t=1/Fs*(0:(Fs-1));
% buf2=buf;


% t=0:length(y);
% length(y)

% Apply the A-weighting filters
% buf = filter(HdAw, buf );


% Plot the results!
% figure(1); plot(t(1:15), buf2(1:15), 'k'); hold on; plot(t(1:15), buf(1:15), 'g'); legend('Linear-weighted Time Record', 'A-weighted Time Record');
% figure(1); plot(t(1:512), x(1:512), 'k'); hold on; plot(t(1:512), y(1:512), 'g'); legend('Linear-weighted Time Record', 'A-weighted Time Record');

% end


% 
% groupArray = [group1,group2,group3];
% 
% index=0;
% 
% countMatrix=[14500,3];
% 
% for SUF = groupArray %[subgroup] %SUF
%     SUF = num2str(SUF);
%     index=index+1;
% 
% %SUF='n';
% run(['sensorLog18Rev_' SUF '.m'])
% 
% filename = ['audio_' SUF '.bin'];
% 
%  end
% 
% 
% 
% average=std(countMatrix ,0,2);
% %average=mean(countMatrix ,2);
% 
% 
% 
% size(average)
% size(Tred)
% 
