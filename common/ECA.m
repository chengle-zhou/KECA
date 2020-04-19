function [sorted_entropy_index,entropy] = ECA(D,E)
    
    N = size(E,2);
    
    % Renyi entropy estimate sorted »Œ“‚Ïÿπ¿º∆≈≈–Ú
    entropy = diag(D)' .* (ones(1,N)*E).^2;
    [sorted_entropy,sorted_entropy_index] = sort(entropy,'descend');