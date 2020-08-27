#include "fe_vector.hpp"

void fe_hex8(double* fe,const double* xe, const double* xw, const unsigned int NGT) {

    // x=[xi, eta, zeta], w=[weight_xi, weight_eta, weight_zeta]
    double x[3], w[3];
    double *N;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double jaco;

    double xp, yp, zp, force;

    for (int i = 0; i < 8; i++) {
        fe[i] = 0.0;
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::HEX8);

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // weight_zeta

                N  = NdN_object.Nvalues(x);
                dN = NdN_object.dNvalues(x);

                // initialize
                for (int p = 0; p < 3; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (int p = 0; p < 3; p++){
                    for (int m = 0; m < 8; m++){
                        dxds[p] += xe[3*m]*dN[3*m + p];
                        dyds[p] += xe[3*m + 1]*dN[3*m + p];
                        dzds[p] += xe[3*m + 2]*dN[3*m + p];
                    }
                }

                // jacobian
                jaco =dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                        dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                if (jaco < 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }

                // physical coordinates of Gauss points
                xp = 0.0;
                yp = 0.0;
                zp = 0.0;
                for (unsigned int n = 0; n < 8; n++){
                    xp += xe[n*3] * N[n];
                    yp += xe[(n*3) + 1] * N[n];
                    zp += xe[(n*3) + 2] * N[n];
                }
                // loading function (USER-DEFINED, ad-hoc at the moment, fixme)
                force = sin(2*M_PI*xp) * sin(2*M_PI*yp) * sin(2*M_PI*zp);

                // fe vector
                for (unsigned int n = 0; n < 8; n++){
                    fe[n] += force * N[n] * jaco * w[2] * w[1] * w[0];
                }

            } // k integration
        } // j integration
    } // i integration
} // fe_hex8



void fe_hex8_eig(Eigen::Matrix<double,8,1> &fe, const double* xe, const double* be, const double* xw, const unsigned int NGT) {

    double x[3], w[3]; // x=[xi,eta,zeta], w=[weight_xi, weight_eta,weight_zeta]
    double *N;
    double *dN;
    double dxds[3], dyds[3], dzds[3];
    double jaco;

    double xp, yp, zp, force;

    for (int i = 0; i < 8; i++) {
        fe(i) = 0.0;
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::HEX8);

    for (int i = 0; i < NGT; i++){
        x[0] = xw[i*2];                     // xi
        w[0] = xw[i*2 + 1];                 // weight_xi
        for (int j = 0; j < NGT; j++){
            x[1] = xw[j*2];                 // eta
            w[1] = xw[j*2 + 1];             // weight_eta
            for (int k = 0; k < NGT; k++){
                x[2] = xw[k*2];             // zeta
                w[2] = xw[k*2 + 1];         // weight_zeta

                N = NdN_object.Nvalues(x);
                dN = NdN_object.dNvalues(x);

                // initialize
                for (int p = 0; p < 3; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (int p = 0; p < 3; p++){
                    for (int m = 0; m < 8; m++){
                        dxds[p] += xe[3*m]*dN[3*m + p];
                        dyds[p] += xe[3*m + 1]*dN[3*m + p];
                        dzds[p] += xe[3*m + 2]*dN[3*m + p];
                    }
                }

                // jacobian
                jaco =dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                      dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                if (jaco < 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }

                // compute element force vector using "approximation" of body force
                for (unsigned int iN = 0; iN < 8; iN++){
                    for (unsigned int jN = 0; jN < 8; jN++){
                        fe(iN) += N[iN] * N[jN] * jaco * w[1] * w[0] * be[jN];
                    }
                }

                /* // compute element force vector using analytic function of body force
                // physical coordinates of Gauss points
                xp = 0.0;
                yp = 0.0;
                zp = 0.0;
                for (unsigned int n = 0; n < 8; n++){
                    xp += xe[n*3] * N[n];
                    yp += xe[(n*3) + 1] * N[n];
                    zp += xe[(n*3) + 2] * N[n];
                }
                // loading function (USER-DEFINED, ad-hoc at the moment, fixme)
                //force = sin(2*M_PI*xp) * sin(2*M_PI*yp) * sin(2*M_PI*zp); // this is for fem3d
                force = 3.0*sin(xp) * sin(yp) * sin(zp);   // this is for fem3da

                // fe vector
                for (unsigned int n = 0; n < 8; n++){
                    fe(n) += force * N[n] * jaco * w[2] * w[1] * w[0];
                } */

            } // k integration
        } // j integration
    } // i integration
} // fe_hex8_eig



void fe_hex8_iso(Eigen::Matrix<double,24,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss){

    const unsigned int NNODES_ELEM = 8; // number of nodes per element
    const unsigned int NDOFS_NODE = 3;  // number of dofs per node
    const unsigned int NDIMS = 3;       // number of physical AND reference coordinates

    // reference coordinates (xi, eta, zeta)
    double x[NDIMS];
    // weight (w_xi, w_eta, w_zeta)
    double w[NDIMS];

    // shape functions evaluated at Gauss points
    double *N;
    // derivatives of shape functions at Gauss points
    double *dN;

    // derivatives of physical coordinates wrt to reference coordinates
    double dxds[NDIMS], dyds[NDIMS], dzds[NDIMS];
    // jacobian
    double jaco;

    for (unsigned int nid = 0; nid < NNODES_ELEM; nid++) {
        for (unsigned int did = 0; did < NDOFS_NODE; did++){
            fe(nid * NDOFS_NODE + did) = 0.0;
        }
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::HEX8);

    for (unsigned int iGauss = 0; iGauss < nGauss; iGauss++){
        x[0] = GaussPoints[iGauss * 2];                     // xi
        w[0] = GaussPoints[iGauss * 2 + 1];                 // weight_xi

        for (unsigned int jGauss = 0; jGauss < nGauss; jGauss++){
            x[1] = GaussPoints[jGauss * 2];                 // eta
            w[1] = GaussPoints[jGauss * 2 + 1];             // weight_eta

            for (unsigned int kGauss = 0; kGauss < nGauss; kGauss++){
                x[2] = GaussPoints[kGauss * 2];             // zeta
                w[2] = GaussPoints[kGauss * 2 + 1];         // weight_zeta

                // get the values of shape functions at Gauss points
                N = NdN_object.Nvalues(x);
                // get the values of derivatives of shape functions at Gauss points
                dN = NdN_object.dNvalues(x);

                // initialize
                for (unsigned int p = 0; p < NDIMS; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (unsigned int n = 0; n < NNODES_ELEM; n++){
                    for (unsigned int p = 0; p < NDIMS; p++){
                        // dx/dxi, dx/deta, dx/dzeta
                        dxds[p] += xe[NDIMS * n] * dN[NDIMS * n + p];
                        // dy/dxi, dy/deta, dy/dzeta
                        dyds[p] += xe[NDIMS * n + 1] * dN[NDIMS * n + p];
                        // dz/dxi, dz/deta, dz/dzeta
                        dzds[p] += xe[NDIMS * n + 2] * dN[NDIMS * n + p];
                    }
                }

                // jacobian
                jaco = dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                      dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                if (jaco <= 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }

                // ah-hoc: compute fe by direct integrating of body force [bx, by, bz]
                /* double rho = 1.0;
                double g = 1.0;
                for (unsigned int nid = 0; nid < NNODES_ELEM; nid++){
                    for (unsigned int did = 0; did < NDOFS_NODE; did++){
                        if (did == 2){
                            fe(nid * NDOFS_NODE + did) += N[nid]*(-rho * g) * jaco * w[2] * w[1] * w[0];
                        } else {
                            fe(nid * NDOFS_NODE + did) = 0.0;
                        }

                    }
                } */

                // compute fe by "approximate" body force bx = sum_{i=1}^{N}(N_i*bx_i), by = ...
                for (unsigned int iDof = 0; iDof < NDOFS_NODE; iDof++){
                    for (unsigned int iN = 0; iN < NNODES_ELEM; iN++){
                        for (unsigned int jN = 0; jN < NNODES_ELEM; jN++){
                            fe(iN * NDOFS_NODE + iDof) += N[iN] * N[jN] * jaco * w[2] * w[1] * w[0] * bN[jN * NDOFS_NODE + iDof];
                        }
                    }
                }

            } // kGauss integration
        } // jGauss integration
    } // iGauss integration
} // fe_hex8_iso



void feT_hex8_iso(Eigen::Matrix<double,12,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss){

    const unsigned int NNODES_FACE = 4; // number of nodes per surface
    const unsigned int NDOFS_NODE = 3;  // number of dofs per node
    const unsigned int NDIMS_PHY = 3;     // number of physical coordinates
    const unsigned int NDIMS_REF = 2;     // number of reference coordinates

    // reference coordinates (xi, eta, zeta)
    double x[NDIMS_REF];
    // weight (w_xi, w_eta, w_zeta)
    double w[NDIMS_REF];

    // shape functions evaluated at Gauss points
    double *N;
    // derivatives of shape functions at Gauss points
    double *dN;

    // derivatives of physical coordinates wrt to reference coordinates
    double dxds[NDIMS_REF], dyds[NDIMS_REF], dzds[NDIMS_REF];
    // jacobian
    double jaco;

    for (unsigned int nid = 0; nid < NNODES_FACE; nid++) {
        for (unsigned int did = 0; did < NDOFS_NODE; did++){
            feT(nid * NDOFS_NODE + did) = 0.0;
        }
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::QUAD4);

    for (unsigned int iGauss = 0; iGauss < nGauss; iGauss++){
        x[0] = GaussPoints[iGauss * 2];                     // xi
        w[0] = GaussPoints[iGauss * 2 + 1];                 // weight_xi

        for (unsigned int jGauss = 0; jGauss < nGauss; jGauss++){
            x[1] = GaussPoints[jGauss * 2];                 // eta
            w[1] = GaussPoints[jGauss * 2 + 1];             // weight_eta

            // get the values of shape functions at Gauss points
            N = NdN_object.Nvalues(x);
            // get the values of derivatives of shape functions at Gauss points
            dN = NdN_object.dNvalues(x);

            // initialize
            for (unsigned int p = 0; p < NDIMS_REF; p++){
                dxds[p] = 0.0;          //[dx/dxi, dx/deta]
                dyds[p] = 0.0;          //[dy/dxi, dy/deta]
                dzds[p] = 0.0;          //[dz/dxi, dz/deta]
            }
            for (unsigned int n = 0; n < NNODES_FACE; n++){
                for (unsigned int p = 0; p < NDIMS_REF; p++){
                    // dx/dxi, dx/deta
                    dxds[p] += xe[NDIMS_PHY * n] * dN[NDIMS_REF * n + p];
                    // dy/dxi, dy/deta
                    dyds[p] += xe[NDIMS_PHY * n + 1] * dN[NDIMS_REF * n + p];
                    // dz/dxi, dz/deta
                    dzds[p] += xe[NDIMS_PHY * n + 2] * dN[NDIMS_REF * n + p];
                }
            }

            // jacobian
            jaco = (dyds[0]*dzds[1] - dyds[1]*dzds[0]) * (dyds[0]*dzds[1] - dyds[1]*dzds[0]) +
                   (dzds[0]*dxds[1] - dzds[1]*dxds[0]) * (dzds[0]*dxds[1] - dzds[1]*dxds[0]) +
                   (dxds[0]*dyds[1] - dxds[1]*dyds[0]) * (dxds[0]*dyds[1] - dxds[1]*dyds[0]);
            jaco = sqrt(jaco);

            for (unsigned int iDof = 0; iDof < NDOFS_NODE; iDof++){
                for (unsigned int iN = 0; iN < NNODES_FACE; iN++){
                    for (unsigned int jN = 0; jN < NNODES_FACE; jN++){
                        feT(iN * NDOFS_NODE + iDof) += N[iN] * N[jN] * jaco * w[1] * w[0] * tN[jN * NDOFS_NODE + iDof];
                    }
                }
            }

        } // jGauss integration
    } // iGauss integration
} // feT_hex8_iso


void fe_hex20_iso(Eigen::Matrix<double,60,1> &fe, const double* xe, const double* bN,
                const double* GaussPoints, const unsigned int nGauss){

    const unsigned int NNODES_ELEM = 20; // number of nodes per element
    const unsigned int NDOFS_NODE = 3;  // number of dofs per node
    const unsigned int NDIMS_PHY = 3;       // number of physical coordinates
    const unsigned int NDIMS_REF = 3;   // number of reference coordinates

    // reference coordinates (xi, eta, zeta)
    double x[NDIMS_REF];
    // weight (w_xi, w_eta, w_zeta)
    double w[NDIMS_REF];

    // shape functions evaluated at Gauss points
    double *N;
    // derivatives of shape functions at Gauss points
    double *dN;

    // derivatives of physical coordinates wrt to reference coordinates
    double dxds[NDIMS_REF], dyds[NDIMS_REF], dzds[NDIMS_REF];
    // jacobian
    double jaco;

    for (unsigned int nid = 0; nid < NNODES_ELEM; nid++) {
        for (unsigned int did = 0; did < NDOFS_NODE; did++){
            fe(nid * NDOFS_NODE + did) = 0.0;
        }
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::HEX20);

    for (unsigned int iGauss = 0; iGauss < nGauss; iGauss++){
        x[0] = GaussPoints[iGauss * 2];                     // xi
        w[0] = GaussPoints[iGauss * 2 + 1];                 // weight_xi

        for (unsigned int jGauss = 0; jGauss < nGauss; jGauss++){
            x[1] = GaussPoints[jGauss * 2];                 // eta
            w[1] = GaussPoints[jGauss * 2 + 1];             // weight_eta

            for (unsigned int kGauss = 0; kGauss < nGauss; kGauss++){
                x[2] = GaussPoints[kGauss * 2];             // zeta
                w[2] = GaussPoints[kGauss * 2 + 1];         // weight_zeta

                // get the values of shape functions at Gauss points
                N = NdN_object.Nvalues(x);
                // get the values of derivatives of shape functions at Gauss points
                dN = NdN_object.dNvalues(x);

                // initialize
                for (unsigned int p = 0; p < NDIMS_REF; p++){
                    dxds[p] = 0.0;          //[dx/dxi, dx/deta, dx/dzeta]
                    dyds[p] = 0.0;          //[dy/dxi, dy/deta, dy/dzeta]
                    dzds[p] = 0.0;          //[dz/dxi, dz/deta, dz/dzeta]
                }
                for (unsigned int n = 0; n < NNODES_ELEM; n++){
                    for (unsigned int p = 0; p < NDIMS_REF; p++){
                        // dx/dxi, dx/deta, dx/dzeta
                        dxds[p] += xe[NDIMS_PHY * n] * dN[NDIMS_REF * n + p];
                        // dy/dxi, dy/deta, dy/dzeta
                        dyds[p] += xe[NDIMS_PHY * n + 1] * dN[NDIMS_REF * n + p];
                        // dz/dxi, dz/deta, dz/dzeta
                        dzds[p] += xe[NDIMS_PHY * n + 2] * dN[NDIMS_REF * n + p];
                    }
                }

                // jacobian
                jaco = dxds[0]*(dyds[1]*dzds[2] - dzds[1]*dyds[2]) + dyds[0]*(dzds[1]*dxds[2] - dxds[1]*dzds[2]) +
                      dzds[0]*(dxds[1]*dyds[2] - dyds[1]*dxds[2]);
                if (jaco <= 0.0) {
                    printf(" Jacobian is negative!");
                    exit(0);
                }

                // ad hoc: compute fe by direct integrating of body force [bx, by, bz]
                /* double rho = 1.0;
                double g = 1.0;
                for (unsigned int nid = 0; nid < NNODES_ELEM; nid++){
                    for (unsigned int did = 0; did < NDOFS_NODE; did++){
                        if (did == 2){
                            fe(nid * NDOFS_NODE + did) += N[nid]*(-rho * g) * jaco * w[2] * w[1] * w[0];
                        } else {
                            fe(nid * NDOFS_NODE + did) = 0.0;
                        }

                    }
                } */

                // compute fe by "approximate" body force bx = sum_{i=1}^{N}(N_i*bx_i), by = ...
                for (unsigned int iDof = 0; iDof < NDOFS_NODE; iDof++){
                    for (unsigned int iN = 0; iN < NNODES_ELEM; iN++){
                        for (unsigned int jN = 0; jN < NNODES_ELEM; jN++){
                            fe(iN * NDOFS_NODE + iDof) += N[iN] * N[jN] * jaco * w[2] * w[1] * w[0] * bN[jN * NDOFS_NODE + iDof];
                        }
                    }
                }

            } // kGauss integration
        } // jGauss integration
    } // iGauss integration
}// fe_hex20_iso


void feT_hex20_iso(Eigen::Matrix<double,24,1> &feT, const double* xe, const double* tN,
                const double* GaussPoints, const unsigned int nGauss){

    const unsigned int NNODES_FACE = 8;   // number of nodes per surface
    const unsigned int NDOFS_NODE = 3;    // number of dofs per node
    const unsigned int NDIMS_PHY = 3;     // number of physical coordinates
    const unsigned int NDIMS_REF = 2;     // number of reference coordinates

    // reference coordinates (xi, eta, zeta)
    double x[NDIMS_REF];
    // weight (w_xi, w_eta, w_zeta)
    double w[NDIMS_REF];

    // shape functions evaluated at Gauss points
    double *N;
    // derivatives of shape functions at Gauss points
    double *dN;

    // derivatives of physical coordinates wrt to reference coordinates
    double dxds[NDIMS_REF], dyds[NDIMS_REF], dzds[NDIMS_REF];
    // jacobian
    double jaco;

    for (unsigned int nid = 0; nid < NNODES_FACE; nid++) {
        for (unsigned int did = 0; did < NDOFS_NODE; did++){
            feT(nid * NDOFS_NODE + did) = 0.0;
        }
    }

    // shape functions and their derivaties
    shape::shapeFunc<double> NdN_object(shape::ELEMTYPE::QUAD8);

    for (unsigned int iGauss = 0; iGauss < nGauss; iGauss++){
        x[0] = GaussPoints[iGauss * 2];                     // xi
        w[0] = GaussPoints[iGauss * 2 + 1];                 // weight_xi

        for (unsigned int jGauss = 0; jGauss < nGauss; jGauss++){
            x[1] = GaussPoints[jGauss * 2];                 // eta
            w[1] = GaussPoints[jGauss * 2 + 1];             // weight_eta

            // get the values of shape functions at Gauss points
            N = NdN_object.Nvalues(x);
            // get the values of derivatives of shape functions at Gauss points
            dN = NdN_object.dNvalues(x);

            // initialize
            for (unsigned int p = 0; p < NDIMS_REF; p++){
                dxds[p] = 0.0;          //[dx/dxi, dx/deta]
                dyds[p] = 0.0;          //[dy/dxi, dy/deta]
                dzds[p] = 0.0;          //[dz/dxi, dz/deta]
            }
            for (unsigned int n = 0; n < NNODES_FACE; n++){
                for (unsigned int p = 0; p < NDIMS_REF; p++){
                    // dx/dxi, dx/deta
                    dxds[p] += xe[NDIMS_PHY * n] * dN[NDIMS_REF * n + p];
                    // dy/dxi, dy/deta
                    dyds[p] += xe[NDIMS_PHY * n + 1] * dN[NDIMS_REF * n + p];
                    // dz/dxi, dz/deta
                    dzds[p] += xe[NDIMS_PHY * n + 2] * dN[NDIMS_REF * n + p];
                }
            }

            // jacobian
            jaco = (dyds[0]*dzds[1] - dyds[1]*dzds[0]) * (dyds[0]*dzds[1] - dyds[1]*dzds[0]) +
                   (dzds[0]*dxds[1] - dzds[1]*dxds[0]) * (dzds[0]*dxds[1] - dzds[1]*dxds[0]) +
                   (dxds[0]*dyds[1] - dxds[1]*dyds[0]) * (dxds[0]*dyds[1] - dxds[1]*dyds[0]);
            jaco = sqrt(jaco);

            for (unsigned int iDof = 0; iDof < NDOFS_NODE; iDof++){
                for (unsigned int iN = 0; iN < NNODES_FACE; iN++){
                    for (unsigned int jN = 0; jN < NNODES_FACE; jN++){
                        feT(iN * NDOFS_NODE + iDof) += N[iN] * N[jN] * jaco * w[1] * w[0] * tN[jN * NDOFS_NODE + iDof];
                    }
                }
            }
        } // jGauss integration
    } // iGauss integration
}