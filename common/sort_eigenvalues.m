function [D,E] = sort_eigenvalues(D,E);
    
    % Ectract eigenvalues from diagonal of D
    d = diag(D);
    
    % Sort the eigenvalues
    [d_sorted d_index] = sort(d,'descend');
    
    % Create new matrix D
    D = zeros(length(d_sorted));
    for i = 1 : length(d_sorted);
        D(i,i) = d_sorted(i);
    end;
    
    % Create new matrix E
    E = E(:,d_index);