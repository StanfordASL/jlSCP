# --------------------------------------------
# -   Plotting script for the dubins example -
# --------------------------------------------

# Python plotting with matplotlib
using PyCall, LaTeXStrings
import PyPlot; const plt = PyPlot

include("../Models/dubins_free.jl")
include("../SCP/scp_problem.jl")


# --------------------
# Plotting with PyPlot
# --------------------
function plt_circle(ax, pos, radius; color="k", alpha=1., label="None")
    # Filled circle
    circle = plt.matplotlib.patches.Circle(pos, radius=radius,
                    color=color, fill=true, alpha=alpha)
    ax.add_patch(circle)
    # Edge of circle, with alpha 1.
    if label=="None"
        c_edge = plt.matplotlib.patches.Circle(pos, radius=radius, 
                        color=color, alpha=1, fill=false)
    else
        c_edge = plt.matplotlib.patches.Circle(pos, radius=radius, 
                        color=color, alpha=1, fill=false, label=label)
    end
    ax.add_patch(c_edge)
    return ax
end

function plt_solutions(scp_problem::SCPProblem, model, X_all, U_all;
                        xlims=[-0.5,3.], ylims=[0.0,6.], figsize=(8,6), B_plot_labels=true)
    N = length(X_all)
    idx = [1,2]

    fig = plt.figure(figsize=figsize)
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(X_all[1][idx[1],:], X_all[1][idx[2],:], "--",
                    label="Initializer", linewidth=3, color="b")
    for iter = 2:length(X_all)#-3
        X = X_all[iter]
        ax.plot(X[idx[1],:], X[idx[2],:], "o--",
                    label="Iterate $(iter - 1)", linewidth=2, markersize=7)
    end

    # initial / final conditions

    plt.scatter(model.x_init[idx[1]], model.x_init[idx[2]], color="black")
    plt.scatter(model.x_final[idx[1]], model.x_final[idx[2]], color="black")

    # Plot obstacles
    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plt_circle(ax, p_obs[idx], obs_radius; color="r", alpha=0.3)
    end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 30
    rcParams["font.family"] = "Helvetica"
    plt.xlim(xlims)
    plt.ylim(ylims)
    if B_plot_labels 
        plt.legend(loc="upper left", fontsize=23.5, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)
    plt.xticks([0,1,2,3])
    plt.yticks([0,1,2,3])

    plt.draw()

    return fig
end

function plt_final_solution(scp_problem::SCPProblem, model, X, U)
    N = length(X_all)
    idx = [1,2]

    fig = plt.figure(figsize=(4.5, 7.5))
    ax  = plt.gca()

    # Plot final solution
    plt.plot(X[idx[1],:], X[idx[2],:], "bo-", 
                linewidth=2, markersize=6)
    plt.plot(Inf*[1,1],Inf*[1,1], "b-", label="Trajectory") # for legend

    # Plot Thrust
    for k = 1:(length(X[1,:])-1)
        xk, uk =  X[:,k], U[:,k]

        uMax, mag = 23.2, 1.5

        plt.arrow(xk[idx[1]], xk[idx[2]], 
                    mag*(uk[idx[1]]/uMax), mag*(uk[idx[2]]/uMax),
                    color="g")
    end
    plt.plot(Inf*[1,1],Inf*[1,1], "g-", label="Thrust") # for legend

    # Plot obstacles
    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plt_circle(ax, p_obs[idx], obs_radius; color="r", alpha=0.4)
    end
    plt.plot(Inf*[1,1],Inf*[1,1], "r-", label="Obstacle") # for legend

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    plt.title("Final Dubins Trajectory", pad=15)
    plt.xlim([-0.5,3.])
    plt.ylim([ 0.0,6.])
    plt.xlabel("E")
    plt.ylabel("N")    
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(alpha=0.3)
    
    plt.draw()

    return fig
end

function plt_final_angle_accel(scp_problem::SCPProblem, model, X, U)
    N = length(X_all)
    t_max = 2.

    times = collect(range(0,stop=(SCPproblem.N-1)*SCPproblem.dt,length=SCPproblem.N))[1:SCPproblem.N-1]
    norms_U = sqrt.(U[1,:].^2+U[2,:].^2+U[3,:].^2)

    fig = plt.figure(figsize=(5.5, 7.5))

    # -------------
    # Tilt Angle
    plt.subplot(2,1,1)
    tilt_angles = U[3,:]./norms_U
    tilt_angles = (180. / pi) * tilt_angles
    plt.plot(times, tilt_angles, "bo-", 
                linewidth=1, markersize=4)
    # max tilt angle
    theta_max = (180/pi) * (pi/3.0)
    plt.plot([0,t_max], theta_max*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], theta_max*ones(2), 90, 
                        color="r", alpha=0.2)
    # Params
    plt.title("Tilt Angle", pad=10)
    plt.xlim([0, t_max])
    plt.ylim([0, 65   ])
    # plt.xlabel("Time [s]")
    plt.ylabel(L"$\theta$ [deg]")
    plt.grid(alpha=0.3)
    plt.draw()

    # -------------
    # Acceleration
    plt.subplot(2,1,2)
    fig.tight_layout(pad=2.0)

    plt.plot(times, norms_U, "bo-", 
                linewidth=1, markersize=4)

    # max/min acceleration
    a_min, a_max = 0.6, 23.2
    plt.plot([0,t_max], a_max*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_max*ones(2), 90, 
                        color="r", alpha=0.2)
    plt.plot([0,t_max], a_min*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_min*ones(2), -5, 
                        color="r", alpha=0.2)

    # Parameters / Settings / Labels
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    plt.title("Cmd. Acceleration", pad=10)
    plt.xlim([0, t_max])
    plt.ylim([-1, 25  ])
    plt.xlabel("Time [s]")
    plt.ylabel(L"$u [m/s^2]$")
    plt.grid(alpha=0.3)
    plt.draw()

    return fig
end

function plt_controls(scp_problem::SCPProblem, model, X_all, U_all;
                        xlims=[-0.5,3.], ylims=[0.0,6.], figsize=(8,6), B_plot_labels=true)
    N = length(X_all)
    idx = [1,2]

    fig = plt.figure(figsize=figsize)
    ax  = plt.gca()

    # Plot SCP solutions
    times = collect(range(0,stop=X_all[1][end,end],length=scp_problem.N))[1:scp_problem.N-1]
    t_max = X_all[1][end,end]
    plt.plot(times, U_all[1][1,:], "--",
                    label="Initializer", linewidth=3, color="b")
    for iter = 2:length(X_all)#-3
        X, U = X_all[iter], U_all[iter]
        times = collect(range(0,stop=X[end,end],length=scp_problem.N))[1:scp_problem.N-1]
        plt.plot(times, U[1,:], "o--",
                        label="Iterate $(iter - 1)", linewidth=2, markersize=7)
        t_max = maximum([t_max, X[end,end]])
    end

    # max/min control
    a_min, a_max = model.uMin[1], model.uMax[1]
    plt.plot([0,t_max], a_max*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_max*ones(2), 0.3, 
                        color="r", alpha=0.2)
    plt.plot([0,t_max], a_min*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_min*ones(2), -0.3, 
                        color="r", alpha=0.2,
                        label="Control bounds")


    # # initial / final conditions

    # plt.scatter(model.x_init[idx[1]], model.x_init[idx[2]], color="black")
    # plt.scatter(model.x_final[idx[1]], model.x_final[idx[2]], color="black")

    # # Plot obstacles
    # for obs_i = 1:length(model.obstacles)
    #     p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
    #     plt_circle(ax, p_obs[idx], obs_radius; color="r", alpha=0.3)
    # end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 30
    rcParams["font.family"] = "Helvetica"
    plt.xlim(xlims)
    plt.ylim(ylims)
    if B_plot_labels 
        plt.legend(bbox_to_anchor=(1, 0.75), fontsize=23.5, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)
    # plt.xticks([0,1,2,3])
    # plt.yticks([0,1,2,3])

    plt.xlim([-0.1,t_max+0.1])
    plt.ylim([-0.32,0.32])

    plt.xlabel(L"$t$")
    plt.ylabel(L"$u$")

    # plt.draw()

    return fig
end