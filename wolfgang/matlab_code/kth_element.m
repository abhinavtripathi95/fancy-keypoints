function el=kth_element(a,n,k)

% kth_element       finds k-th smallest element in n-list a
%
% el=kth_element(a,n,k)
%

l=1;
m=n;
while l<m
    x=a(k);
    i=l;
    j=m;
    for ii=1:n
        while (a(i) < x) i=i+1;end;
        while (x < a(j)) j=j-1;end;
            if i <= j
                t= a(i);
                a(i)=a(j);
                a(j)=t;
                i=i+1;
                j=j-1;
            end;    
        if i > j break;end;      
    end;    
    if j<k l=i; end;
    if k<i m=j; end;
end;    
el=a(k);
return;