include("../Models/dubins_free.jl")
include("./scp_problem.jl")

function solve_scp(model, scp_problem, N, max_it; B_shooting=true, verbose=false)
    # Defining SCP parameters
    (Delta0, omega0, omegamax, epsilon,
            convergence_threshold) = get_initial_scp_parameters(model)

    Xp,Up        = initialize_trajectory(model,N)
    X, U         = copy(Xp), copy(Up)
    omega, Delta = omega0, Delta0
    X_all, U_all, P0_all = [], [], []
    push!(X_all, copy(X))
    push!(U_all, copy(U))

    # SCP loop
    success, it        = false, 1
    B_shooting_success = false
    while !(B_shooting_success) &&
          !(success && 
            (convergence_metric(model, X_all[end-2],U_all[end-2],Xp,Up) +
             convergence_metric(model, X,U,Xp,Up) ) < convergence_threshold) &&
          (omega < omegamax) && 
          (it < max_it)
        if verbose
            println("-----------\nIteration $it\n-----------")
        end

        # Storing the solution at the previous step and the linearized dynamics
        Xp, Up                    = copy(X), copy(U)
        model.f, model.A, model.B = compute_dynamics(model,Xp,Up)
        
        # Defining the convex subproblem
        reset_problem(     scp_problem, model)
        set_parameters(    scp_problem, model, Xp, Up, omega, Delta)
        define_nonconvex_cost(scp_problem, model)
        define_constraints(scp_problem, model)
        
        # Solving the convex subproblem
        JuMP.optimize!(scp_problem.solver_model)
        X_sol, U_sol = JuMP.value.(scp_problem.X), JuMP.value.(scp_problem.U)
        p0_dual = get_initial_constraint_dual_variable(scp_problem, model)
        push!(P0_all,copy(p0_dual))
        # println("tf=",X_sol[end,end])       

        # -----------
        # Shrink trust regions
        if verbose
            println("Accept solution.")
        end
        X, U    = copy(X_sol), copy(U_sol)
        Delta   = 0.95 * Delta
        if satisfies_trust_region_constraints(scp_problem, model, X, U, Xp, Up)
            success = true
        else
            success = false
        end

        # --------
        # Shooting
        if B_shooting
            p0_dual = -get_initial_constraint_dual_variable(scp_problem, model)
            tf_guess= X_sol[end,end]
            if verbose
                println("it $it: trying shooting with tf=",tf_guess)#, ", p0=", p0_dual)
            end
            B_RK4, B_HYBRD = true, false
            Xs, Us, Ps, tf, converged = shooting_solve(p0_dual, tf_guess, N, model, 
                                                        B_RK4=B_RK4, B_HYBRD=B_HYBRD, B_verbose=verbose)
            # println("tf=",tf, ", p0=", Ps[:,1], "converged=", converged)
            if converged && tf>0
                if verbose
                    println("tf=",tf, ", converged=", converged)
                end
                success = true
                B_shooting_success = true
                X, U   = Xs, Us
                Xp, Up = Xs, Us
            end
        end


        # Collecting the solution at each iteration
        push!(X_all,copy(X))
        push!(U_all,copy(U))
        it += 1
        
        if !B_shooting_success
            if (it == max_it) || # reached max nb. of iterations
               (it < 3)          # needs at least 3 iterations to check convergence
                success = false
            end
        end

        if verbose
            println("Parameters:")
            println("omega  = $omega; Delta = $Delta")
            println("metric = $(convergence_metric(model,X,U,Xp,Up))")
            println("success= $success")
        end
    end
    if verbose
        println(">>> Finished <<<")
    end

    X_f,U_f,X_fp,U_fp = X_all[end],U_all[end],X_all[end-1],U_all[end-1]

    B_trust_satisfied = satisfies_trust_region_constraints(scp_problem, model, X_f,U_f,X_fp,U_fp)
    if !B_trust_satisfied && verbose
        println(">>>>> UNSATISFIED trust region constraint !!! ")#Satisfies trust region constraint.")
    end

    return (X_all,U_all,P0_all, success, it, B_trust_satisfied)
end