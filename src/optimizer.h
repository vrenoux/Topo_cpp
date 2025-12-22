#pragma once

#include <vector>
#include <cmath>

#include "material.h"
#include "mesh_core.h"
#include "boundary.h"

struct Penalization
{
    double p = 3.0;

    double get_penalized_density(double density) const {
        return std::pow(density, p);
    }

    double get_penalization_derivative(double density) const {
        if (density <= 0.0) return 0.0;
        return p * std::pow(density, p - 1);
    }
};

class Optimizer {
public:
    // Constructor
    Optimizer() = default;
    explicit Optimizer(int m) : max_iter(m) {}

    // Destructor
    ~Optimizer() {};

    //Setters
    void setMaxIter(int m) { max_iter = m; }

    void setVolumeFractionConstraint(double vfc) { volume_fraction_constraint = vfc; }

    void setNDesignVariables(int num) { n = num; }

    void setDesignVariableBounds(const std::vector<double>& xmin_in,
                                 const std::vector<double>& xmax_in) {
        xmin = xmin_in;
        xmax = xmax_in;
    }

    void setPenalization(Penalization pen) { penalization = pen; }

    void Optimize(const fem::LinearElasticityMaterial& mat,
                  msh::Mesh& M,
                  const fem::BoundaryConditions& bcs);
    

private:
    int max_iter=100;
    int iter = 0;
    int n = 0;

    double volume_fraction_constraint = 0.5;

    std::vector<double> xmin;
    std::vector<double> xmax;

    Penalization penalization;

};


