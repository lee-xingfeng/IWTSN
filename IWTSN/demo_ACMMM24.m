clear;
% clc;

addpath(genpath('./'));



datadir='Datasets/';
Dataname = cell(13 , 1);        
Dataname{2} = 'ORL_mtv_m';

numdata = length(Dataname); % number of the test datasets

for idata = [2]
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
        datafile = [datadir, cell2mat(Dataname(idata)),'.mat'];
        load(datafile);
        filename = [cell2mat(Dataname(idata)),'.mat'];
         %% -----------------data preparation...-----------------
        cls_num = length(unique(Y));
        k= cls_num;
        sample_num = length(Y);
        numview = length(X);
        tic;
        if(size(Y,2)~=1)
              Y = Y';
        end
        if ~isempty(find(Y==0,1))
                Y = Y + 1;
        end
        for v = 1:numview
         if size(X{v},2)~=sample_num
               X{v} = X{v}';
         end
         X{v} = NormalizeFea(X{v},0);
        end


        d = [2*k];
        numanchor = 2*k;
%% -----------------data preparation...-----------------        
if strcmp('ORL_mtv_m.mat',filename)
     TempLambda1 = 0.9;%
     TempLambda2 = 2.^[-1];
     TempLambda3 = 2.^[-3]; 
end
       
        
        res_all = [];
        time1 = toc;
        maxAcc = 0;
        

        savedata = [];
        idx = 1;
        for LambdaIndex1 = 1 : length(TempLambda1)
            p = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                alpha = TempLambda2(LambdaIndex2);
                for LambdaIndex3 = 1 : length(TempLambda3)
                    gamma = TempLambda3(LambdaIndex3);
                disp([char(Dataname(idata)),'-l1=', num2str(p), '-l2=', num2str(alpha),'-l3=', num2str(gamma)]);
                tic; 
                [F,WZ,B,Q,Z,S,iter,obj,ts,X_complete] = ACMMM24(X,Y,d,numanchor,alpha,p,gamma); % X,Y,lambda,d,numanchor  
                F = F ./ (repmat(sqrt(sum(F .^ 2, 2)), 1, k)); 
                time2 = toc;
                stream = RandStream.getGlobalStream;
                reset(stream);
                MAXiter = 1000; 
                REPlic = 20; 
                tic;
                for rep = 1 : 20
                    pY = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep, : ) = Clustering8Measure(Y, pY);
                end
                mean_res = mean(res)
                mean_std = std(res);
                savedata = [savedata; p, alpha, gamma,mean_res,mean_std];
                time3 = toc;
                runtime(idx) = time1 + time2 + time3/20;
                disp(['runtime:', num2str(runtime(idx))])            
                end
            end
        end

end

