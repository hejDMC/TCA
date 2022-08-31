function similarity_tca(F1,F2,omega)
    # inputs: F1 and F2 are two TCA models, 
    #         omega are all the permutations of R components
    # output the model similarity of the two TCA models
    # ref : Williams et al., Neuron 2018

    # get the number of components from the models
    R = length(F1.lambdas)
    sm = 0.0 #set as float not integer
    # loop to sum over the components
    for r = 1:R
        # get the permutation indices
        r2 = omega[r]
        # first factor with thte lamdas terms of the tca models
        x = (1 - abs(F1.lambdas[r]-F2.lambdas[r2])/max(F1.lambdas[r],F2.lambdas[r2]))
            #inner dot products of the columns to obtain the angles of the planes
            for f =1:3
                a = F1.factors[f][:,r]
                b = F2.factors[f][:,r2]
                x *= a'*b/norm(a)/norm(b); # scale to the unit of the the factors with norm as the TCA function dosen't do it 
            end
        sm += x       
    end
    return sm/R
end

function similarity_tca(F1,F2)
    # get the number of components from the models and check that they are the same
    R1 = length(F1.lambdas)
    R2 = length(F2.lambdas)
        if R1<R2 || R1>R2
            println("The two TCA models don't have the same number of components")    
        else
            R = length(F1.lambdas)  
            maximum([similarity_tca(F1,F2,omega) for omega = permutations(1:R)])
        end
end
    