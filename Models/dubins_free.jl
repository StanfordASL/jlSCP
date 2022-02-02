# -----------------------------------------------------------
# -   Model for the Dubins car with free final time example -
# -----------------------------------------------------------

using DifferentialEquations # not necessary is RungeKutta is used for shooting method (default)
using MINPACK
using NLsolve

using Distributions
using Random

export DubinsFree



# Model DubinsFree as a Julia class

mutable struct DubinsFree
    # State (x,y,theta,tf) and control (u) dimensions
    x_dim
    u_dim

    # Dynamics and linearized dynamics
    f
    A
    B

    # Problem settings
    x_init
    x_final
    tf_guess
    tf
    xMin
    xMax
    uMin
    uMax
    v
    k

    B_free_final_time
    B_free_final_angle

    # Cylindrical obstacles (modeled by a center (x,y) and a radius r) and polygon obstacles (not used in this example)
    eps_obstacles
    obstacles
    poly_obstacles
    alpha

    # SCP parameters
    Delta0
    omega0
    omegamax
    # threshold for constraints satisfaction : constraints <= epsilon
    epsilon
    epsilon_xf_constraint
    convergence_threshold # in %
end



# The problem is set in the class constructor

function DubinsFree()
    x_dim = 4 # (last component is final time, only last index is used.)
    u_dim = 1

    v = 1.
    k = 2.
    B_free_final_time  = true # ALWAYS TRUE
    B_free_final_angle = false

    x_init  = [ 0.; 0; 0]
    x_final = [ 2.; 3; 0]
    tf_guess = 3. # guess
    tf       = 1. # normalized time for dynamics constraints discretization
    myInf = 1.0e6 # Adopted to detect initial and final condition-free state variables
    xMin = [-100., -100., -2*pi]
    xMax = [ 100.,  100.,  2*pi]
    uMin = [-0.25]
    uMax = [ 0.25]
    true_cost_weight = 1.

    Delta0 = 3.
    omega0 = 5000.
    omegamax = 1.0e6
    epsilon = 1e-3
    epsilon_xf_constraint = 0.
    convergence_threshold = 0.001

    # Obstacles
    eps_obstacles = 0.05
    obstacles = []
    obs = [[0.8,2],0.5]
    push!(obstacles, obs)
    alpha = 5000.

    poly_obstacles = []

    DubinsFree(x_dim, u_dim,
             [], [], [],
             x_init, x_final, 
             tf_guess, tf, 
             xMin, xMax,
             uMin, uMax,
             v, k,
             B_free_final_time, B_free_final_angle,
             eps_obstacles,
             obstacles,
             poly_obstacles,
             alpha,
             Delta0,
             omega0,
             omegamax,
             epsilon,
             epsilon_xf_constraint,
             convergence_threshold)
end



# Method that returns the SCP parameters (used for set up)

function get_initial_scp_parameters(m::DubinsFree)
    return m.Delta0, m.omega0, m.omegamax, m.epsilon, m.convergence_threshold
end



# SCP is intialized by zero controls and velocities, and a straight-line in position

function initialize_trajectory(model::DubinsFree, N::Int)
  x_dim,  u_dim   = model.x_dim, model.u_dim
  x_init, x_final = model.x_init, model.x_final
  
  X        = zeros(x_dim, N)
  X[1:3,:] = hcat(range(x_init, stop=x_final, length=N)...)
  X[end,:] .= model.tf_guess

  U = zeros(u_dim, N-1) .+ 1e-3

  return X, U
end



# Method that returns the convergence ratio between iterations (in percentage)
# The quantities X, U denote the actual solution over time, whereas Xp, Up denote the solution at the previous step over time

function convergence_metric(model::DubinsFree, X, U, Xp, Up)
    x_dim = model.x_dim
    N = length(X[1,:])

    dt = model.tf/(N-1)
    Delta_uks = 0.
    for k in 1:(N-1)
        Delta_uks = Delta_uks + dt*(norm(U[:,k] - Up[:,k]))^2
    end
    return Delta_uks
end



# Method that returns the original cost

function true_NL_cost(model::DubinsFree, solver_model, X, U, Xp, Up)
    x_dim, u_dim     = model.x_dim, model.u_dim
    N = length(U[1,:])+1

    if model.B_free_final_time
        @NLexpression(solver_model, trueNLcost, (Xp[end,end]/(N-1)) * sum(U[i,k]^2 for k in 1:length(U[1,:]) for i in 1:u_dim))
    else
        @NLexpression(solver_model, trueNLcost, (model.tf/(N-1)) * sum(U[i,k]^2 for k in 1:length(U[1,:]) for i in 1:u_dim))
    end

    return trueNLcost
end



# The following methods return the i-th coordinate at the k-th iteration of the various constraints and their linearized versions (when needed)
# These are returned in the form " g(t,x(t),u(t)) <= 0 "



# Method that gathers all the linear control constraints

function control_max_convex_constraints(model::DubinsFree, X, U, Xp, Up, k, i)
    return ( U[i, k] - model.uMax[i] )
end

function control_min_convex_constraints(model::DubinsFree, X, U, Xp, Up, k, i)
    return ( model.uMin[i] - U[i, k] )
end


# State bounds and trust-region constraints (these are all convex constraints)

function state_max_convex_constraints(model::DubinsFree, X, U, Xp, Up, k, i)
    return ( X[i, k] - model.xMax[i] )
end



function state_min_convex_constraints(model::DubinsFree, X, U, Xp, Up, k, i)
    return ( model.xMin[i] - X[i, k] )
end



function trust_region_max_constraints(model::DubinsFree, X, U, Xp, Up, k, i, Delta)
    return ( (X[i, k] - Xp[i, k]) - Delta )
end



function trust_region_min_constraints(model::DubinsFree, X, U, Xp, Up, k, i, Delta)
    return ( -Delta - (X[i, k] - Xp[i, k]) )
end



# Method that checks whether trust-region constraints are satisifed or not (recall that trust-region constraints are penalized)

function is_in_trust_region(model::DubinsFree, X, U, Xp, Up, Delta)
    B_is_inside = true

    for k = 1:length(X[1,:])
        for i = 1:model.x_dim
            if trust_region_max_constraints(model, X, U, Xp, Up, k, i, Delta) > 0.
                B_is_inside = false
            end
            if trust_region_min_constraints(model, X, U, Xp, Up, k, i, Delta) > 0.
                B_is_inside = false
            end
        end
    end

    return B_is_inside
end


# Initial and final conditions on state variables

function state_initial_constraints(model::DubinsFree, X, U, Xp, Up)
    return ( X[1:3,1] - model.x_init )
end



function state_final_constraints(model::DubinsFree, X, U, Xp, Up)
    if model.B_free_final_angle
      return ( X[1:2,end] - model.x_final[1:2] )
    else
      return ( X[1:3,end] - model.x_final )
    end
end



# Methods that return the cylindrical obstacle-avoidance constraint and its lienarized version
# Here, a merely classical distance function is considered


function obstacle_constraint(model::DubinsFree, X, U, Xp, Up, k, obs_i,
                                               obs_type::String="sphere")
    #  obs_type    : Type of obstacles, can be 'sphere' or 'poly'
    if obs_type=="sphere"
      dim_obs           = 2
      p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
      p_k               = X[1:dim_obs,k]
      
      dist       = norm(p_k - p_obs, 2)
      constraint = -( dist - obs_radius )
    else
      print("[DubinsFree.jl::obstacle_constraint_convexified] Unknown obstacle type.")
    end
    return constraint
end
function obstacle_constraint_convexified(model::DubinsFree, X, U, Xp, Up, k, obs_i,
                                                           obs_type::String="sphere")
    #  obs_type    : Type of obstacles, can be 'sphere' or 'poly'
    if obs_type=="sphere"
      dim_obs           = 2
      p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
      p_k               = X[1:dim_obs,k]
      p_kp              = Xp[1:dim_obs,k]


      dist_prev  = norm(p_kp - p_obs, 2)
      dir_prev   = (p_kp - p_obs)/dist_prev
      constraint = -( dist_prev - obs_radius + sum(dir_prev[i] * (p_k[i] - p_kp[i]) for i=1:dim_obs) )
    else
      print("[DubinsFree.jl::obstacle_constraint_convexified] Unknown obstacle type.")
    end
    return constraint
end




# The following methods return the dynamical constraints and their linearized versions
# These are returned as time-discretized versions of the constraints " x' - f(x,u) = 0 " or " x' - A(t)*(x - xp) - B(t)*(u - up) = 0 ", respectively



# Method that returns the dynamical constraints and their linearized versions all at once

function compute_dynamics(model, Xp, Up)
    N = length(Xp[1,:])

    f_all, A_all, B_all = [], [], []

    for k in 1:N-1
        x_k = Xp[:,k]
        u_k = Up[:,k]

        f_dyn_k, A_dyn_k, B_dyn_k = f_dyn(x_k, u_k, model), A_dyn(x_k, u_k, model), B_dyn(x_k, u_k, model)

        push!(f_all, f_dyn_k)
        push!(A_all, A_dyn_k)
        push!(B_all, B_dyn_k)
    end

    return f_all, A_all, B_all
end



# These methods return the dynamics and its linearizations with respect to the state (matrix A(t)) and the control (matrix B(t)), respectively

function f_dyn(x::Vector, u::Vector, model::DubinsFree)
  x_dim = model.x_dim
  tf    = x[end]

  f = zeros(x_dim)
  f[1] = tf*model.v*cos(x[3])
  f[2] = tf*model.v*sin(x[3])
  f[3] = tf*model.k*u[1]
  f[4] = 0. # final time

  return f
end

function A_dyn(x::Vector, u::Vector, model::DubinsFree)
  x_dim = model.x_dim
  tf    = x[end]

  A      = zeros(x_dim,x_dim)
  A[1,3] = tf * (-model.v*sin(x[3]))
  A[2,3] = tf * ( model.v*cos(x[3]))

  A[1,4] = model.v*cos(x[3])
  A[2,4] = model.v*sin(x[3])
  A[3,4] = model.k*u[1]

  return A
end

function B_dyn(x::Vector, u::Vector, model::DubinsFree)
  x_dim, u_dim = model.x_dim, model.u_dim
  tf           = x[end]

  B      = zeros(x_dim,u_dim)
  B[:,1] = [0; 0; tf*model.k; 0]
  return B
end







function obstacle_potentialSphere_penalty(model::DubinsFree, 
                                          pos, pos_obs, radius, alpha)
    pad = model.eps_obstacles
    if norm(pos-pos_obs)^2 < (radius+pad)^2
      return alpha * (norm(pos-pos_obs)^2-(radius+pad)^2)^2
    else
      return 0.
    end
end
function obstacle_potentialSphere_penalty_grad(model::DubinsFree, 
                                               g,
                                               pos, pos_obs, radius, alpha)
    pad = model.eps_obstacles
    if norm(pos-pos_obs)^2 < (radius+pad)^2
      g[1] = alpha * 2. * (norm(pos-pos_obs)^2-(radius+pad)^2) * 2. * (pos-pos_obs)[1] 
      g[2] = alpha * 2. * (norm(pos-pos_obs)^2-(radius+pad)^2) * 2. * (pos-pos_obs)[2] 
    else
      g[1] = 0.
      g[2] = 0.
    end
    return g
end
function define_obs_potential_jump_NL_functions(model::DubinsFree, solver_model, 
                                                obs_type::String="sphere")
    if obs_type != "sphere"      
      throw(MethodError("[DubinsFree.jl::define_obs_potential_jump_NL_functions] Only spheres are supported."))
    end
    alpha = model.alpha

    # First obstacle
    pos_obs1, radius1 = model.obstacles[1][1], model.obstacles[1][2]
    obsPotFunc1     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs1, radius1, alpha)
    obsPotFuncGrad1 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs1, radius1, alpha)
    register(solver_model, :obsPotFunc1, 2, obsPotFunc1, obsPotFuncGrad1)

    # 2nd obstacle
    pos_obs2, radius2 = model.obstacles[2][1], model.obstacles[2][2]
    obsPotFunc2     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs2, radius2, alpha)
    obsPotFuncGrad2 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs2, radius2, alpha)
    register(solver_model, :obsPotFunc2, 2, obsPotFunc2, obsPotFuncGrad2)

    # # 3-6 obstacles
    # pos_obs3, radius3 = model.obstacles[3][1], model.obstacles[3][2]
    # obsPotFunc3     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs3, radius3, alpha)
    # obsPotFuncGrad3 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs3, radius3, alpha)
    # register(solver_model, :obsPotFunc3, 2, obsPotFunc3, obsPotFuncGrad3)
    # pos_obs4, radius4 = model.obstacles[4][1], model.obstacles[4][2]
    # obsPotFunc4     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs4, radius4, alpha)
    # obsPotFuncGrad4 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs4, radius4, alpha)
    # register(solver_model, :obsPotFunc4, 2, obsPotFunc4, obsPotFuncGrad4)
    # pos_obs5, radius5 = model.obstacles[5][1], model.obstacles[5][2]
    # obsPotFunc5     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs5, radius5, alpha)
    # obsPotFuncGrad5 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs5, radius5, alpha)
    # register(solver_model, :obsPotFunc5, 2, obsPotFunc5, obsPotFuncGrad5)
    # pos_obs6, radius6 = model.obstacles[6][1], model.obstacles[6][2]
    # obsPotFunc6     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs6, radius6, alpha)
    # obsPotFuncGrad6 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs6, radius6, alpha)
    # register(solver_model, :obsPotFunc6, 2, obsPotFunc6, obsPotFuncGrad6)
end
function obstacle_potential_penalties(model::DubinsFree, solver_model, 
                                    X, U, Xp, Up, 
                                    obs_i,
                                    obs_type::String="sphere")
    N, Nb_obstacles = length(X[1,:]), length(model.obstacles)
    if Nb_obstacles != 1 && Nb_obstacles != 2 && Nb_obstacles != 6
      throw(MethodError("[DubinsFree.jl::obstacle_potential_penalties] Incorrect number of obstacles."))
    end
    if !(obs_type=="sphere")
      print("[DubinsFree.jl::obstacle_potential_penalties] Unknown obstacle type.")
    end

    if !(model.B_free_final_time)
      print("[DubinsFree.jl::obstacle_potential_penalties] tf should be free.")
    end

    if obs_i == 1
      @NLexpression(solver_model, obscost1, (X[end,end]/(N-1))*sum(obsPotFunc1(X[1,k],X[2,k]) for k in 1:N))
      return obscost1
    elseif obs_i == 2
      @NLexpression(solver_model, obscost2, (X[end,end]/(N-1))*sum(obsPotFunc2(X[1,k],X[2,k]) for k in 1:N))
      return obscost2
    # elseif obs_i == 3
    #   @NLexpression(solver_model, obscost3, (X[end,end]/(N-1))*sum(obsPotFunc3(X[1,k],X[2,k]) for k in 1:N))
    #   return obscost3
    # elseif obs_i == 4
    #   @NLexpression(solver_model, obscost4, (X[end,end]/(N-1))*sum(obsPotFunc4(X[1,k],X[2,k]) for k in 1:N))
    #   return obscost4
    # elseif obs_i == 5
    #   @NLexpression(solver_model, obscost5, (X[end,end]/(N-1))*sum(obsPotFunc5(X[1,k],X[2,k]) for k in 1:N))
    #   return obscost5
    # elseif obs_i == 6
    #   @NLexpression(solver_model, obscost6, (X[end,end]/(N-1))*sum(obsPotFunc6(X[1,k],X[2,k]) for k in 1:N))
    #   return obscost6
    else
      print("obs_i=$obs_i")
      throw(MethodError("[DubinsFree.jl::obstacle_potential_penalty] Incorrect number of obstacles."))
    end
end












##### ----------------------------- #####
##### FUNCTIONS FOR SHOOTING METHOD #####



function shooting_solve(p0::Vector, tf_guess::Float64,
                        N::Int, model::DubinsFree; 
                        tol_newton=1e-1, tol_hybrd = 1e-6,
                        B_RK4::Bool=false, B_HYBRD::Bool=true, B_verbose=true)
  x_dim, u_dim = model.x_dim, model.u_dim
  x0, xF       = model.x_init, model.x_final

  # Set up shooting function
  if !B_RK4
    shooting_eval! = (F, p_0_tf) -> parameterized_shooting_eval!(F,
                                                            x0, xF, p_0_tf[1:3], p_0_tf[4],
                                                            model)
  else
    N_RK4shooting = N
    shooting_eval! = (F, p_0_tf) -> parameterized_shooting_eval_RK4!(F,
                                                            x0, xF, p_0_tf[1:3], p_0_tf[4],
                                                            model, N_RK4shooting)
  end

  if !B_HYBRD
    # Run Newton method
    sol_newton = nlsolve(shooting_eval!, [p0;tf_guess], iterations=10, ftol=tol_newton)
    # sol_newton = nlsolve(shooting_eval!, [p0;tf_guess], ftol=tol_newton)
    p0_tf_solution = sol_newton.zero
    if B_verbose
      println(sol_newton)
      println(p0_tf_solution)
    end
  else
    # run Advanced modified version of Powell's algorithm
    # sol_minpack = fsolve(shooting_eval!, [p0;tf_guess], iterations=100, tol=tol_hybrd, method=:hybrd)
    sol_minpack = fsolve(shooting_eval!, [p0;tf_guess], tol=tol_hybrd, method=:hybrd)
    p0_tf_solution = sol_minpack.x
    if B_verbose
      println(sol_minpack)
      println("p0_tf=",p0_tf_solution)
    end
  end

  if (!B_HYBRD && sol_newton.f_converged) || (B_HYBRD && sol_minpack.converged)
    xp_0  = [x0; p0_tf_solution[1:3]; p0_tf_solution[4]]


    # check tolerance (again)
    F = zeros(x_dim)
    shooting_eval!(F, p0_tf_solution)
    if (!B_HYBRD && (norm(sum(F)) > 10*x_dim*tol_newton)) ||
       (B_HYBRD  && (norm(sum(F)) > 10*x_dim*tol_hybrd))
      if B_verbose
        println("Shooting method DID NOT converge ! (F=$(F)))")
      end
      return [],[],[],-1,false
    end
    # check that time did not jump too much
    if norm(p0_tf_solution[4]-tf_guess)>3.
      if B_verbose
        println("Shooting method didnt converge - delta tf > 3.")
      end
      return [],[],[],-1,false
    end
    
    # Solve again to get trajectory
    if !B_RK4
      tspan = (0., p0_tf_solution[4])

      shooting_ode! = (f, u, p, t) -> dynamics_shooting!(f, u[1:3], u[4:6], u[7], model)
      prob = ODEProblem(shooting_ode!, xp_0, tspan)

      dt  = p0_tf_solution[4] / (N-1)

      sol = DifferentialEquations.solve(prob, saveat=dt)#, dtmin=0.1, force_dtmin=true)
      xp   = hcat(sol.u...)

      Xtraj, Ptraj, tf = xp[1:3,:], xp[4:6,:], xp[end,end]
    else
      shooting_ode! = (f, u, t) -> dynamics_shooting!(f, u[1:3], u[4:6], u[7], model)
      t_vec, traj_u = solveRK4(shooting_ode!, model, N, p0_tf_solution[4], xp_0)
      Xtraj, Ptraj, tf = traj_u[1:3,:], traj_u[4:6,:], t_vec[end]
    end
    Xtraj = vcat(Xtraj,tf*ones(1,N))



    Utraj = zeros(u_dim, N-1)
    for k in 1:(N-1)
      Utraj[1,k] = ( Ptraj[3,k]*model.k / 2.)
    end
    return Xtraj, Utraj, Ptraj, tf, true
  else
    if B_verbose
      println("Shooting method DID NOT converge !")
    end
    return [],[],[],-1,false
  end

end


function parameterized_shooting_eval!(F::Vector, 
                                      x0::Vector, xF::Vector, p0::Vector, tf_guess::Float64, 
                                      model::DubinsFree)
  v, k = model.v, model.k

  xp_0  = [x0; p0; tf_guess]
  tspan = (0., tf_guess)

  shooting_ode! = (f, u, p, t) -> dynamics_shooting!(f, u[1:3], u[4:6], u[7], model)
  prob = ODEProblem(shooting_ode!, xp_0, tspan)
  sol  = DifferentialEquations.solve(prob)

  for i = 1:model.x_dim
    if (i == 1) || (i == 2)
      F[i] = (sol.u[end][i]-xF[i])
    elseif i == 3
      if model.B_free_final_angle
        F[i] = sol.u[end][6] # p_theta(t_f) = 0.
      else
        F[i] = sqrt((cos(xF[i])-cos(sol.u[end][i]))^2 + (sin(xF[i])-sin(sol.u[end][i]))^2)
      end
    elseif i == 4
      tf = tf_guess
      xf = sol.u[end][1:3]
      pf = sol.u[end][4:6]
      uf = (pf[3]*k/2.)
      ff = [v * cos(xf[3]), v * sin(xf[3]), k * uf] #f_dyn(xf, [uf], model)
      Hf = sum(pf[i]*ff[i] for i in 1:3) - uf^2

      # Obstacles
      for obs_i in 1:length(model.obstacles)
        pos_obs, radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        Hf = Hf - obstacle_potentialSphere_penalty(model, [traj_end[1], traj_end[2]], 
                                                          pos_obs, radius, model.alpha)
      end
      F[i] = Hf
    else
      println("DubinsFree doesnt have so many dimensions!!!")
    end
  end
end


function parameterized_shooting_eval_RK4!(F::Vector, 
                                      x0::Vector, xF::Vector, p0::Vector, tf_guess::Float64, 
                                      model::DubinsFree, N::Int)
  v, k = model.v, model.k

  xp_0  = [x0; p0; tf_guess]
  tspan = (0., tf_guess)

  shooting_ode! = (f, u, t) -> dynamics_shooting!(f, u[1:3], u[4:6], u[7], model)
  t_vec, traj_u = solveRK4(shooting_ode!, model, N, tf_guess, xp_0)
  # print("traj_u=", size(traj_u))
  traj_end = traj_u[:,end]

  for i = 1:model.x_dim
    if (i == 1) || (i == 2)
      F[i] = traj_end[i]-xF[i]
    elseif i == 3
      if model.B_free_final_angle
        F[i] = traj_end[6] # p_theta(t_f) = 0.
      else
        F[i] = (cos(xF[i])-cos(traj_end[i]))^2 + (sin(xF[i])-sin(traj_end[i]))^2
      end
    elseif i == 4
      tf = tf_guess
      xf = traj_end[1:3]
      pf = traj_end[4:6]
      uf = (pf[3]*k/2.)
      if uf>model.uMax[1]
        uf=model.uMax[1]
      elseif uf<model.uMin[1]
        uf=model.uMin[1]
      end
      ff = [v * cos(xf[3]), v * sin(xf[3]), k * uf] #f_dyn(xf, [uf], model)
      Hf = sum(pf[i]*ff[i] for i in 1:3) - uf^2

      # Obstacles
      for obs_i in 1:length(model.obstacles)
        pos_obs, radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        Hf = Hf - obstacle_potentialSphere_penalty(model, [traj_end[1], traj_end[2]], 
                                                          pos_obs, radius, model.alpha)
      end

      F[i] = Hf
    else
      println("DubinsFree doesnt have so many dimensions!!!")
    end
  end
end

function solveRK4(dynamics::Function,model::DubinsFree, N, tf, z0, t0=0.0)
    # dynamics : (f,x,t) -> populates f=f(x) (= x_dot) 

    x_dim = length(z0)

    h = (tf - t0)/(N - 1)
    t = zeros(N)
    z = zeros(x_dim,N)
    dynTemp = zeros(x_dim)
    zTemp  = zeros(x_dim)
    t[1]   = t0
    z[:,1] = z0

    for i = 1:N-1
        t[i+1] = t[i] + h

        dynamics(dynTemp,z[:,i],t[i])
        z[:,i+1] = z[:,i] + h*dynTemp/6.0

        zTemp = z[:,i] + h*dynTemp/2.0
        dynamics(dynTemp,zTemp,t[i]+h/2.0)
        z[:,i+1] = z[:,i+1] + h*dynTemp/3.0

        zTemp = z[:,i] + h*dynTemp/2.0
        dynamics(dynTemp,zTemp,t[i]+h/2.0)
        z[:,i+1] = z[:,i+1] + h*dynTemp/3.0

        zTemp = z[:,i] + h*dynTemp
        dynamics(dynTemp,zTemp,t[i]+h)
        z[:,i+1] = z[:,i+1] + h*dynTemp/6.0
    end

    return t, z
end

function dynamics_shooting!(f::Vector, x::Vector, p::Vector, tf::Float64, 
                            model::DubinsFree)
    v, k = model.v, model.k
    uopt = ( p[3]*k / 2.)
    if uopt>model.uMax[1]
      uopt=model.uMax[1]
    elseif uopt<model.uMin[1]
      uopt=model.uMin[1]
    end

    f[1] = v * cos(x[3])
    f[2] = v * sin(x[3])
    f[3] = k * uopt
    f[4] = 0.
    f[5] = 0.
    f[6] = v * (sin(x[3])*p[1] - cos(x[3])*p[2])
    f[7] = 0.

    # Obstacles
    for obs_i in 1:length(model.obstacles)
      g               = zeros(2)
      pos_obs, radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
      grads_obs       = obstacle_potentialSphere_penalty_grad(model, g, [x[1], x[2]], 
                                                              pos_obs, radius, model.alpha)
      f[4]            = f[4] + grads_obs[1]
      f[5]            = f[5] + grads_obs[2]
    end
end













# randomized Monte-Carlo experiments
function get_randomized_conditions(model::DubinsFree; N_MC::Int=10, NB_obs::Int=1, seed=12345)
  rng = MersenneTwister(seed)


  # Boundary Conditions
  min_pos, max_pos = -1., 1.
  min_θ0f, max_θ0f = -pi, pi

  x0s = [transpose(min_pos .+ (max_pos-min_pos) .* rand(rng, N_MC));
         transpose(min_pos .+ (max_pos-min_pos) .* rand(rng, N_MC));
         transpose(min_θ0f .+ (max_θ0f-min_θ0f) .* rand(rng, N_MC))]
  x0s = transpose(x0s) # (N_MC x x_dim)

  # Sample final states so they are far enough from each other
  angles    = -pi .+ 2*pi * rand(rng, N_MC)
  dist_x0fs = 4.  .+ 3    * rand(rng, N_MC)
  xfs = [transpose(dist_x0fs .* cos.(angles));
        transpose(dist_x0fs .* sin.(angles));
        transpose(min_θ0f .+ (max_θ0f-min_θ0f) .* rand(rng, N_MC))]
  xfs = transpose(xfs) + x0s

  # set final and initial angles within a cone
  delta_x0fs = xfs-x0s
  for i = 1:N_MC
    theta = atan(delta_x0fs[i,2], delta_x0fs[i,1])
    min_θ0f, max_θ0f = -pi/4+theta, pi/4+theta
    x0s[i,3] = min_θ0f + (max_θ0f-min_θ0f) .* rand(rng)
    xfs[i,3] = min_θ0f + (max_θ0f-min_θ0f) .* rand(rng)
  end

  # Obstacles
  rads = 0.4*ones(N_MC)

  obstacles_vec = []
  for i=1:N_MC
    obss = []
    for j=1:NB_obs
      r    = rads[i]

      # place obs randomly in between
      min_xb, max_xb = min(x0s[i,1], xfs[i,1])+2*r, max(x0s[i,1], xfs[i,1])-2*r
      min_yb, max_yb = min(x0s[i,2], xfs[i,2])+2*r, max(x0s[i,2], xfs[i,2])-2*r
      if min_xb>max_xb
          min_xb_old, max_xb_old = min_xb, max_xb
          min_xb = (min_xb_old+max_xb_old)/2
          max_xb = (min_xb_old+max_xb_old)/2
      end
      if min_yb>max_yb
          min_yb_old, max_yb_old = min_yb, max_yb
          min_yb = (min_yb_old+max_yb_old)/2
          max_yb = (min_yb_old+max_yb_old)/2
      end
      o_x = min_xb+(max_xb-min_xb).*rand(rng)
      o_y = min_yb+(max_yb-min_yb).*rand(rng)
      o   = [o_x,o_y]


      # Check that obstacle is some distance appart from other obstacles
      if j>1
        op = obss[end][1]
        if norm(o-op)<5*r
          o = op + (o-op)/norm(o-op) * 5*r
          o_x,o_y = o[1], o[2]
        end
      end

      # push obstacles so they are 8r from x0 and xf
      if norm(o-x0s[i,1:2]) < 8*r
        o = x0s[i,1:2] + (o-x0s[i,1:2])/norm(o-x0s[i,1:2]) * 8*r
        o_x,o_y = o[1], o[2]
      end
      if norm(o-xfs[i,1:2]) < 8*r
        o = xfs[i,1:2] + (o-xfs[i,1:2])/norm(o-xfs[i,1:2]) * 8*r
        o_x,o_y = o[1], o[2]
      end

      # Again: Check that obstacle is some distance appart from other obstacles
      if j>1
        op = obss[end][1]
        if norm(o-op)<3*r
          o = op + (o-op)/norm(o-op) * 3*r
          o_x,o_y = o[1], o[2]
        end
      end

      # save a single obstacle
      obs = [[o_x, o_y], rads[i]]
      push!(obss, obs)
    end
    push!(obstacles_vec, obss)
  end

  # final times
  # min_tf, max_tf = 0.5, 2.
  min_tf, max_tf = 4, 6.
  tfs = min_tf .+ (max_tf-min_tf) .* rand(rng, N_MC)

  return x0s, xfs, obstacles_vec, tfs
end