function [UU,temp_WZ,P,Q,Z,S,iter,obj2,ts,X] = align_W_local_b(X,gt,d,numanchor,alpha,p,gamma)
%
% alpha: alpha\|QX-P\|s_ij
%beta:tr()
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.

% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations
m = numanchor;
numclass = length(unique(gt));
numview = length(X);
numsample = size(gt,1);

  
%% initialize W,P,Z,E,F
for i = 1:numview
    P{i} = zeros(d,m); 
   di = size(X{i},1); 
   Q{i} = zeros(di,d);
   %initialize
   Z{i}=zeros(m,numsample);% m  * n
   H{i}=zeros(m,numsample);% m  * n
   J{i}=zeros(m,numsample);% m  * n
end

opt.disp = 0;

mu = 10e-5; max_mu = 10e10; pho_mu = 2;
flag = 1;
iter = 0;

%%
while flag
    iter = iter + 1; 
        %% optimize Z ok
    for v=1:numview
        DisQXP{v} = L2_distance_1(P{v},Q{v}'*X{v});
        temp_am{v} = (alpha+0.5*mu)*ones(1,numsample); 
        G{v} = H{v} - J{v}/mu;
        temp_Z{v} = 0.5*mu*G{v}-0.5*gamma*DisQXP{v};
         for ii=1:numsample
            idx = 1:numanchor;
            Z_column = temp_Z{v}(idx,ii)./(temp_am{v}(ii)); %
            Z{v}(idx,ii) = EProjSimplex_new(Z_column');
         end
    end

%      %% optimize H һ��
         for i = 1:numview
             TZ{i} = Z{i}';%Z1
             Y{i} = J{i}';            
         end
    Z_tensor = cat(3, TZ{:,:});
    J_tensor = cat(3, Y{:,:});

%% tensor leanring 2 %%
    Ten = Z_tensor+J_tensor/mu;
    shiftTen=shiftdim(Ten, 1);
%     [H_tensor,~,~] = prox_tnn(shiftTen, 1/mu);
%     [H_tensor,~,~] = prox_n_itnn(shiftTen,1/mu);
    [H_tensor,~,~] = prox_n_w_itnn(shiftTen,shiftTen,1/mu,p);
%      [H_tensor,~,~] = proxy_scahtten(shiftTen,1/mu,p);
%     [H_tensor] = prox_n_itnn_wshrinkObj(shiftTen, Ten, 1/mu);
    H_tensor = shiftdim(H_tensor, 2);
    
    %% optimize Q{v} һ��
    for v = 1:numview
        temp_Q = X{v}*Z{v}'*P{v}';      
        [U,~,V] = svd(temp_Q,'econ');
        Q{v} = U*V';
    end

    %% optimize P{v} 
    for v = 1:numview
    temp_P = Q{v}' * X{v} * Z{v}';
    [Unew,~,Vnew] = svd(temp_P,'econ');
    P{v} = Unew*Vnew';
    end

     %% solve  Y_tensor and  penalty parameters        
    J_tensor = J_tensor + mu*(Z_tensor - H_tensor);
    mu = min(mu*pho_mu, max_mu);
    for v = 1:numview
        H{v} = H_tensor(:,:,v)';
        J{v} = J_tensor(:,:,v)';
    end
    %% obtain indicator from Z
    S=0;
    for v = 1:numview
            S = S + Z{v};
           temp_WZ{v} = Z{v}; 
    end
    S = S/numview;
    [UU,~,V]=svd(S','econ');
    ts{iter} = UU(:,1:numclass);

    %%
        term_ZHtensor = 0;
    for v = 1:numview
        term_ZHtensor = term_ZHtensor +  norm(Z{v} - H{v},'fro')^2;
    end
    obj(iter) = term_ZHtensor;
    if (iter>1)
            obj2(iter) = abs((obj(iter-1)-obj(iter))/(obj(iter-1)));
    end
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(S','econ');
        UU = UU(:,1:numclass);
        flag = 0;
    end
end



    
    
         
         
    
