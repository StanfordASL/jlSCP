# ----------------------------
# -   SCP solver environment -
# ----------------------------

using JuMP


export SCPProblem



# SCP solver as a Julia class

mutable struct SCPProblem
    # Number of time-discretization steps and step-size, respectively
    N
    dt

    # Penalization weight ω and trsut-region constraints radius Δ, respectively
    omega
    Delta

    # Number of linear control constraints and second-order cone control constraints, respectively
    dimLinearConstraintsU
    dimSecondOrderConeConstraintsU

    # Model class
    solver_model

    # Current trajectory X and control U
    X
    U

    # Trajectory X and control U at the previous iteration
    Xp
    Up

    # The intial constraints are defined in the model class (used for set up)
    initial_constraint
end



# Standard constructor

function SCPProblem(model, N, Xp, Up, solver=Ipopt.Optimizer)
    N     = N
    dt    = model.tf / (N-1)
    omega = model.omega0
    Delta = model.Delta0

    dimLinearConstraintsU          = model.dimLinearConstraintsU
    dimSecondOrderConeConstraintsU = model.dimSecondOrderConeConstraintsU

    solver_model = Model(with_optimizer(Ipopt.Optimizer, print_level=0, max_iter=1000, tol=1e-13))
    X = @variable(solver_model, X[1:model.x_dim,1:N  ])
    U = @variable(solver_model, U[1:model.u_dim,1:N-1])

    SCPProblem(N, dt,
                 omega, Delta, 
                 dimLinearConstraintsU, dimSecondOrderConeConstraintsU,
                 solver_model,
                 X, U, Xp, Up,
                 [])
end



# Methods that define the convex subproblem at each new SCP iteration

function reset_problem(scp_problem::SCPProblem, model, solver=Ipopt.Optimizer)
    scp_problem.solver_model = Model(with_optimizer(solver, print_level=0, max_iter=1000, tol=1e-13))
    N = scp_problem.N
    X = @variable(scp_problem.solver_model, X[1:model.x_dim,1:N  ])
    U = @variable(scp_problem.solver_model, U[1:model.u_dim,1:N-1])
    scp_problem.X = X
    scp_problem.U = U

    define_obs_potential_jump_NL_functions(model, scp_problem.solver_model)
end

function set_parameters(scp_problem::SCPProblem, model,
                        Xp, Up, omega, Delta)
    scp_problem.Xp = Xp
    scp_problem.Up = Up
    scp_problem.omega = omega
    scp_problem.Delta = Delta
end

function get_initial_constraint_dual_variable(scp_problem::SCPProblem, model)
    x_dim = model.x_dim
    p0 = zeros(length(scp_problem.initial_constraint))
    for i = 1:length(scp_problem.initial_constraint)
        p0[i] = JuMP.dual(scp_problem.initial_constraint[i])
    end
    return p0
end


# The following methods define the convex subproblem at each iteration in the "JuMP" framework



# Methods that define the cost

function define_nonconvex_cost(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    # Control Cost
    trueNLcost = true_NL_cost(model, solver_model, X, U, Xp, Up)
    # @NLobjective(scp_problem.solver_model, Min, trueNLcost)

    obs1_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 1, "sphere")
    obs2_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 2, "sphere")

    obs3_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 3, "sphere")
    obs4_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 4, "sphere")
    obs5_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 5, "sphere")
    obs6_penalty = obstacle_potential_penalties(model, solver_model,
                                                X, U, Xp, Up, 6, "sphere")

    # @NLobjective(scp_problem.solver_model, Min, trueNLcost + obs1_penalty) 
    # @NLobjective(scp_problem.solver_model, Min, trueNLcost + 
    #                                             obs1_penalty + obs2_penalty)
    @NLobjective(scp_problem.solver_model, Min, trueNLcost + 
                                                obs1_penalty + obs2_penalty + 
                                                obs3_penalty + obs4_penalty + obs5_penalty + obs6_penalty)
end



# Method that checks whether penalized state constraints are hardly satisfied (up to the threshold ε)

function satisfies_state_inequality_constraints(scp_problem::SCPProblem, model, X, U, Xp, Up, Delta)
    B_satisfies_constraints = true
    x_dim = model.x_dim
    N = scp_problem.N
    epsilon = model.epsilon

    # Contribution of trust-region constraints
    for k = 1:N
        for i = 1:x_dim
            constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta)
            constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta)
            if constraint_max > epsilon || constraint_min > epsilon
                print("[scp_problem.jl] - trust_region_constraint violated at i=$i and k=$k\n")
                B_satisfies_constraints = false
            end
        end
    end

    # Contribution of obstacle-avoidance constraints
    for k = 1:N
        # Non-polygonal obstacles
        Nb_obstacles = length(model.obstacles)
        if Nb_obstacles > 0
            for i = 1:Nb_obstacles
                constraint = obstacle_constraint(model, X, U, [], [], k, i,
                                                        "sphere")
                if constraint > epsilon
                    print("[scp_problem.jl] - obstacle_constraint violated at i=$i and k=$k, value=$constraint\n")
                    B_satisfies_constraints = false
                end
            end
        end

        # Polygonal obstacles
        Nb_poly_obstacles = length(model.poly_obstacles)
        if Nb_poly_obstacles > 0
            for i = 1:Nb_poly_obstacles
                # constraint = poly_obstacle_constraint(model, X, U, [], [], k, i)
                constraint = obstacle_constraint(model, X, U, Xp, Up, k, i,
                                                        "poly")
                if constraint > epsilon
                    print("[scp_problem.jl] - poly_obstacles_constraint violated at i=$i and k=$k\n")
                    B_satisfies_constraints = false
                end
            end
        end
    end

    # Contribution of state constraints that are different from trust-region constraints and obstacle-avoidance constraints
    for k = 1:N
        for i = 1:x_dim
            constraint_max = state_max_convex_constraints(model, X, U, [], [], k, i)
            constraint_min = state_min_convex_constraints(model, X, U, [], [], k, i)
            if constraint_max > epsilon || constraint_min > epsilon
                print("[scp_problem.jl] - state_max_convex_constraints violated at i=$i and k=$k\n")
                B_satisfies_constraints = false
            end
        end
    end

    return B_satisfies_constraints
end



# These methods add the remaining constraints, such as linearized dyamical, intial/final conditions constraints and control constraints



function define_constraints(scp_problem::SCPProblem, model)
    add_initial_constraints(scp_problem, model)
    add_final_constraints(scp_problem, model)
    add_dynamics_constraints(scp_problem, model)
    add_trust_region_constraints(scp_problem, model)
end



function add_initial_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    constraint = state_initial_constraints(model, X, U, Xp, Up)
    scp_problem.initial_constraint = @constraint(solver_model, constraint .== 0.)
end



# To improve robustness, final conditions are imposed up to a threshold ε > 0 
#   (not always necessary, but can be depending on dynamics and discretization scheme to enable reachability to xf)
function add_final_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta
    epsilon      = model.epsilon_xf_constraint
    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up

    constraint = state_final_constraints(model, X, U, Xp, Up)
    @constraint(solver_model,  constraint .== 0.)
    # @constraint(solver_model,  constraint - epsilon .<= 0.)
    # @constraint(solver_model, -constraint - epsilon .<= 0.)
end



function add_dynamics_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    N, dt        = length(X[1,:]), scp_problem.dt

    # Euler discretization scheme
    # for k = 1:N-1
    #     X_knext  = X[:, k+1]
    #     X_k      = X[:, k]
    #     U_k      = U[:, k]
    #     X_kp     = Xp[:, k]
    #     U_kp     = Up[:, k]
    #     f_dyn_kp = model.f[k]
    #     A_kp     = model.A[k]
    #     B_kp     = model.B[k]

    #     constraint =  X_knext - ( X_k + dt * (  f_dyn_kp + 
    #                                             A_kp * (X_k-X_kp) + 
    #                                             B_kp * (U_k-U_kp)
    #                                          )
    #                             )
    #     @constraint(solver_model, constraint .== 0.)
    # end

    # Trapezoidal discretization scheme
    for k = 1:N-1
        X_kn,  X_k  = X[:,k+1],  X[:,k]
        X_kp        = Xp[:,k]
        U_k,   U_kp = U[:,k],    Up[:,k]
        f_dyn_kp, A_kp, B_kp = model.f[k], model.A[k], model.B[k]

        if k<N-1
            X_knp        = Xp[:,k+1]
            U_kn,  U_knp = U[:,k+1], Up[:,k+1]
            f_dyn_knp    = model.f[k+1]
            A_knp        = model.A[k+1]
            B_knp        = model.B[k+1]
            constraint   = (X_kn-X_k) - 0.5*dt * (
                                 (f_dyn_kp  + A_kp *(X_k-X_kp)   + B_kp *(U_k-U_kp) ) + 
                                 (f_dyn_knp + A_knp*(X_kn-X_knp) + B_knp*(U_kn-U_knp))
                                                 )
        else
            constraint =  (X_kn-X_k) - dt * (f_dyn_kp + A_kp *(X_k-X_kp) + B_kp *(U_k-U_kp) ) 
        end
        @constraint(solver_model, constraint .== 0.)
    end
end



function add_trust_region_constraints(scp_problem::SCPProblem, model)

    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    Delta        = scp_problem.Delta

    X, U, Xp, Up = scp_problem.X, scp_problem.U, scp_problem.Xp, scp_problem.Up
    N, dt        = length(X[1,:]), scp_problem.dt

    for k = 1:N
        for i = 1:x_dim
            constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta)
            constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta)
            @constraint(solver_model, constraint_max <= 0.)
            @constraint(solver_model, constraint_min <= 0.)
        end
    end
    if model.B_free_final_time
        constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, N, x_dim, Delta)
        constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, N, x_dim, Delta)
        @constraint(solver_model, constraint_max <= 0.)
        @constraint(solver_model, constraint_min <= 0.)
    end
end


function satisfies_trust_region_constraints(scp_problem::SCPProblem, model,
                                            X,U, Xp,Up)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    Delta        = scp_problem.Delta

    N, dt        = length(X[1,:]), scp_problem.dt

    B_satisfied = true

    for k = 1:N
        for i = 1:x_dim
            constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta)
            constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta)
            if constraint_min>=0 || constraint_max>=0
                B_satisfied = false
            end
        end
    end
    if model.B_free_final_time
        constraint_max = trust_region_max_constraints(model, X, U, Xp, Up, N, x_dim, Delta)
        constraint_min = trust_region_min_constraints(model, X, U, Xp, Up, N, x_dim, Delta)
        if constraint_min>=0 || constraint_max>=0
            B_satisfied = false
        end
    end

    return B_satisfied
end
