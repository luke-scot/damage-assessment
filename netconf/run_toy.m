function run_toy()
    edges = csvread('toy/edges.csv');
    display(edges)
    priors = csvread('toy/dpriors.csv');
    adj_mat = edges_to_adjmat(edges);
    H = [1,0;0,1];
    B = netConf(adj_mat,H,priors);
    csvwrite('toy/dbeliefs.csv',B);
    fprintf('Output written to toy/dbeliefs.csv\n');
end

function adj_mat = edges_to_adjmat(edges)
    N = max(max(edges));
    edges = unique([edges;edges(:,[2,1])],'rows');
    edges = edges(edges(:,1)~=edges(:,2),:);
    adj_mat = sparse(edges(:,1),edges(:,2),1,N,N);
end