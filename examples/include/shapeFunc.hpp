/**
 * @file shapeFunc.hpp
 * @author Hari Sundar   hsundar@gmail.com
 * @author Han Duc Tran  hantran@cs.utah.edu
 *
 * @brief class for shape functions and derivatives
 *
 * @version 0.1
 * @date 2020-02-29
 */

#ifndef ADAPTIVEMATRIX_SHAPEFUNC_H
#define ADAPTIVEMATRIX_SHAPEFUNC_H

#include <iostream>

namespace shape {

    enum class ELEMTYPE {TRI3, TRI6, QUAD4, QUAD8, HEX8, HEX20, TET4, TET10, HEX27};
    enum class ERROR {UNKNOWN_ELEMENT_TYPE, SUCCESS};

    template<typename DT>
    class shapeFunc {
        private:
        ELEMTYPE elemType;
        unsigned int numNodes;

        public:
        DT* N;  // shape functions
        DT* dN; // derivatives of shape functions

        public:
        shapeFunc(ELEMTYPE eltype);
        ~shapeFunc();

        static unsigned int nNodesPerElem(shape::ELEMTYPE etype) {
            switch (etype) {
                case shape::ELEMTYPE::TRI3: return 3; break;
                case shape::ELEMTYPE::TRI6: return 6; break;
                case shape::ELEMTYPE::QUAD4: return 4; break;
                case shape::ELEMTYPE::QUAD8: return 8; break;
                case shape::ELEMTYPE::HEX8: return 8; break;
                case shape::ELEMTYPE::HEX20: return 20; break;
                case shape::ELEMTYPE::TET4: return 4; break;
                case shape::ELEMTYPE::TET10: return 10; break;
                case shape::ELEMTYPE::HEX27: return 27; break; // 2022.09.01, copied from aMat_dev
                default:
                    return (unsigned int)shape::ERROR::UNKNOWN_ELEMENT_TYPE;
            };
            return (unsigned int)shape::ERROR::UNKNOWN_ELEMENT_TYPE;
        }

        /**
         * @brief: compute values of shape functions at Gauss points
         * @param[in] x natural coordinates
         * @param[out] pointer to N values
        * */
        DT* Nvalues(const DT xi []);

        /**
         * @brief: compute values of derivatives of shape functions at Gauss points
         * @param[in] x natural coordinates
         * @param[out] pointer to dN values
        * */
        DT* dNvalues(const DT xi []);

        // return values of quadratic Lagrange interpolation function at x
        double LQuad(double x, unsigned int idx) {
            if (idx == 0) {
                return (0.5*x*(x - 1.0));
            } else if (idx == 1) {
                return (1.0 - x*x);
            } else if (idx == 2) {
                return (0.5*x*(x+1.0));
            } else {
                std::cout << "LQuad: index of quadratic Lagrange interpolation function is out of range\n";
                exit(1);
            }
        }

        // return values of derivative of quadratic Lagrange interpolation function at x
        double DLQuad(double x, unsigned int idx) {
            if (idx == 0) {
                return (x - 0.5);
            } else if (idx == 1) {
                return (-2.0*x);
            } else if (idx == 2) {
                return (x + 0.5);
            } else {
                std::cout << "DLQuad: index of quadratic Lagrange interpolation function is out of range\n";
                exit(1);
            }
        }
    };

    template <typename DT>
    shapeFunc<DT>::shapeFunc(ELEMTYPE eltype){
        
        elemType = eltype;
        
        numNodes = nNodesPerElem(eltype);

        // allocate memory for N and dN
        if (numNodes > 0){
            N = new DT [numNodes];
            if ((elemType == shape::ELEMTYPE::TRI3) ||
                (elemType == shape::ELEMTYPE::TRI6) ||
                (elemType == shape::ELEMTYPE::QUAD4) || 
                (elemType == shape::ELEMTYPE::QUAD8)){
                // 2D elements: linear or quadratic triangle, linear or quadratic rectangle
                // dN[]/dxi and dN[]/deta
                dN = new DT [ 2 * numNodes ];
            } else if ((elemType == shape::ELEMTYPE::HEX8) || 
                    (elemType == shape::ELEMTYPE::HEX20) ||
                    (elemType == shape::ELEMTYPE::TET4) ||
                    (elemType == shape::ELEMTYPE::TET10) ||
                    (elemType == shape::ELEMTYPE::HEX27)){
                // 3D elements: linear or quadratic tetrahedron, linear or quadratic hexahedron
                // dN[]/dxi, dN[]/deta and dN[]/dzeta
                dN = new DT [ 3 * numNodes ];
            }
        } else {
            N = nullptr;
            dN = nullptr;
        }
    }
    template <typename DT>
    shapeFunc<DT>::~shapeFunc(){
        if (N != nullptr) delete [] N;
        if (dN != nullptr) delete [] dN;
    }

    template <typename DT>
    DT* shapeFunc<DT>::Nvalues(const DT xi[]){
        int xij, etaj, zetaj;

        if (elemType == ELEMTYPE::HEX8){
            // linear 8-node hex element
            N[0] = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) / 8.0;
            N[1] = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 - xi[2]) / 8.0;
            N[2] = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) / 8.0;
            N[3] = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 - xi[2]) / 8.0;
            N[4] = (1.0 - xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) / 8.0;
            N[5] = (1.0 + xi[0]) * (1.0 - xi[1]) * (1.0 + xi[2]) / 8.0;
            N[6] = (1.0 + xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) / 8.0;
            N[7] = (1.0 - xi[0]) * (1.0 + xi[1]) * (1.0 + xi[2]) / 8.0;
        } else if (elemType == ELEMTYPE::QUAD4){
            // linear 4-node quad element
            N[0] = 0.25 * (1.0 - xi[0]) * (1.0 - xi[1]);
            N[1] = 0.25 * (1.0 + xi[0]) * (1.0 - xi[1]);
            N[2] = 0.25 * (1.0 + xi[0]) * (1.0 + xi[1]);
            N[3] = 0.25 * (1.0 - xi[0]) * (1.0 + xi[1]);
        } else if (elemType == ELEMTYPE::QUAD8){
            N[0] = (-0.25) * (1 - xi[0])*(1 - xi[1])*(1 + xi[0] + xi[1]);
            N[1] = (-0.25) * (1 + xi[0])*(1 - xi[1])*(1 - xi[0] + xi[1]);
            N[2] = (-0.25) * (1 + xi[0])*(1 + xi[1])*(1 - xi[0] - xi[1]);
            N[3] = (-0.25) * (1 - xi[0])*(1 + xi[1])*(1 + xi[0] - xi[1]);
            N[4] = 0.5 * (1 - xi[0]*xi[0])*(1 - xi[1]);
            N[5] = 0.5 * (1 - xi[1]*xi[1])*(1 + xi[0]);
            N[6] = 0.5 * (1 - xi[0]*xi[0])*(1 + xi[1]);
            N[7] = 0.5 * (1 - xi[1]*xi[1])*(1 - xi[0]);
        } else if (elemType == ELEMTYPE::TRI3){
            N[0] = 1.0 - xi[0] - xi[1];
            N[1] = xi[0];
            N[2] = xi[1];
        } else if (elemType == ELEMTYPE::TRI6){
            N[0] = 2.*(1.0-xi[0]-xi[1])*(1.0-xi[0]-xi[1]- .5);
            N[1] = 2.*xi[0]*(xi[0]- .5);
            N[2] = 2.*xi[1]*(xi[1]- .5);
            N[3] = 4.*(1.0-xi[0]-xi[1])*xi[0];
            N[4] = 4.*xi[0]*xi[1];
            N[5] = 4.*xi[1]*(1.0-xi[0]-xi[1]);
        } else if (elemType == ELEMTYPE::TET4){
            N[0] = 1.0 - xi[0] - xi[1] - xi[2];
            N[1] = xi[0];
            N[2] = xi[1];
            N[3] = xi[2];
        } else if (elemType == ELEMTYPE::TET10){
            // shape functions are given in Appendix A5 in the book
            // "The Finite element method for solid and structural mechanics", Zienkiewicz
            // here, we apply for reference element
            const DT L0 = 1.0 - xi[0] - xi[1] - xi[2];
            const DT L1 = xi[0];
            const DT L2 = xi[1];
            const DT L3 = xi[2];

            N[0] = L0 * ((2.0 * L0) - 1.0);
            N[1] = L1 * ((2.0 * L1) - 1.0);
            N[2] = L2 * ((2.0 * L2) - 1.0);
            N[3] = L3 * ((2.0 * L3) - 1.0);

            N[4] = 4.0 * L0 * L1;
            N[5] = 4.0 * L1 * L2;
            N[6] = 4.0 * L2 * L0;
            
            N[7] = 4.0 * L0 * L3;
            N[8] = 4.0 * L1 * L3;
            N[9] = 4.0 * L2 * L3;

        } else if (elemType == ELEMTYPE::HEX20){
            // 20-node quadratic serendipity element
            for (unsigned int j = 0; j < 8; j++){
                if (j == 0){
                    xij = -1; etaj = -1; zetaj = -1;
                } else if (j == 1){
                    xij = 1; etaj = -1; zetaj = -1;
                } else if (j == 2){
                    xij = 1; etaj = 1; zetaj = -1;
                } else if (j == 3){
                    xij = -1; etaj = 1; zetaj = -1;
                } else if (j == 4){
                    xij = -1; etaj = -1; zetaj = 1;
                } else if (j == 5){
                    xij = 1; etaj = -1; zetaj = 1;
                } else if (j == 6){
                    xij = 1; etaj = 1; zetaj = 1;
                } else if (j == 7){
                    xij = -1; etaj = 1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                N[j] = 0.125 * (1 + xij*xi[0])*(1 + etaj*xi[1])*(1 + zetaj*xi[2])*(xij*xi[0] + etaj*xi[1] + zetaj*xi[2] - 2);
            }
            for (unsigned int j = 8; j < 15; j += 2){
                if (j == 8){
                    etaj = -1; zetaj = -1;
                } else if (j == 10){
                    etaj = 1; zetaj = -1;
                } else if (j == 12){
                    etaj = -1; zetaj = 1;
                } else if (j == 14){
                    etaj = 1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                N[j] = 0.25 * (1 - xi[0]*xi[0])*(1 + etaj*xi[1])*(1 + zetaj*xi[2]);
            }
            for (unsigned int j = 9; j < 16; j += 2){
                if (j == 9){
                    xij = 1; zetaj = -1;
                } else if (j == 11){
                    xij = -1; zetaj = -1;
                } else if (j == 13){
                    xij = 1; zetaj = 1;
                } else if (j == 15){
                    xij = -1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                N[j] = 0.25 * (1 - xi[1]*xi[1])*(1 + xij*xi[0])*(1 + zetaj*xi[2]);
            }
            for (unsigned int j = 16; j < 20; j++){
                if (j == 16){
                    xij = -1; etaj = -1;
                } else if (j == 17){
                    xij = 1; etaj = -1;
                } else if (j == 18){
                    xij = 1; etaj = 1;
                } else if (j == 19){
                    xij = -1; etaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                N[j] = 0.25 * (1 - xi[2]*xi[2])*(1 + xij*xi[0])*(1 + etaj*xi[1]);
            }
        } else if (elemType == ELEMTYPE::HEX27){
            const DT L0xi = LQuad(xi[0], 0);
            const DT L1xi = LQuad(xi[0], 1);
            const DT L2xi = LQuad(xi[0], 2);

            const DT L0eta = LQuad(xi[1], 0);
            const DT L1eta = LQuad(xi[1], 1);
            const DT L2eta = LQuad(xi[1], 2);

            const DT L0zeta = LQuad(xi[2], 0);
            const DT L1zeta = LQuad(xi[2], 1);
            const DT L2zeta = LQuad(xi[2], 2);
            // this is to follow the node ordering of Gmsh: http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_65.php#SEC65
            // 8 corner nodes:
            N[0] = L0xi * L0eta * L0zeta;
            N[1] = L2xi * L0eta * L0zeta;
            N[2] = L2xi * L2eta * L0zeta;
            N[3] = L0xi * L2eta * L0zeta;
            N[4] = L0xi * L0eta * L2zeta;
            N[5] = L2xi * L0eta * L2zeta;
            N[6] = L2xi * L2eta * L2zeta;
            N[7] = L0xi * L2eta * L2zeta;
            // 12 mid-side edge nodes:
            N[8] = L1xi * L0eta * L0zeta;
            N[9] = L0xi * L1eta * L0zeta;
            N[10] = L0xi * L0eta * L1zeta;
            N[11] = L2xi * L1eta * L0zeta;
            N[12] = L2xi * L0eta * L1zeta;
            N[13] = L1xi * L2eta * L0zeta;
            N[14] = L2xi * L2eta * L1zeta;
            N[15] = L0xi * L2eta * L1zeta;
            N[16] = L1xi * L0eta * L2zeta;
            N[17] = L0xi * L1eta * L2zeta;
            N[18] = L2xi * L1eta * L2zeta;
            N[19] = L1xi * L2eta * L2zeta;
            // 6 face-center nodes
            N[20] = L1xi * L1eta * L0zeta;
            N[21] = L1xi * L0eta * L1zeta;
            N[22] = L0xi * L1eta * L1zeta;
            N[23] = L2xi * L1eta * L1zeta;
            N[24] = L1xi * L2eta * L1zeta;
            N[25] = L1xi * L1eta * L2zeta;
            // 1 center node
            N[26] = L1xi * L1eta * L1zeta;
        } else {
            std::cout << "Not implemented yet\n";
            exit(0);
        }
        return N;
    }

    template<typename DT>
    DT* shapeFunc<DT>::dNvalues(const DT xi []){
        int xij, etaj, zetaj;
        
        if (elemType == ELEMTYPE::HEX8){
            // linear 8-node hex element
            // dN[0]/dxi, dN[0]/deta, dN[0]/dzeta
            dN[0] = (1.0 - xi[1]) * (1.0 - xi[2]) / (-8.0);
            dN[1] = (1.0 - xi[0]) * (1.0 - xi[2]) / (-8.0);
            dN[2] = (1.0 - xi[0]) * (1.0 - xi[1]) / (-8.0);

            // dN[1]/dxi, dN[1]/deta, dN[1]/dzeta
            dN[3] = (1.0 - xi[1]) * (1.0 - xi[2]) / (8.0);
            dN[4] = (1.0 + xi[0]) * (1.0 - xi[2]) / (-8.0);
            dN[5] = (1.0 + xi[0]) * (1.0 - xi[1]) / (-8.0);

            // dN[2]/dxi, dN[2]/deta, dN[2]/dzeta
            dN[6] = (1.0 + xi[1]) * (1.0 - xi[2]) / (8.0);
            dN[7] = (1.0 + xi[0]) * (1.0 - xi[2]) / (8.0);
            dN[8] = (1.0 + xi[0]) * (1.0 + xi[1]) / (-8.0);

            // dN[3]/dxi, dN[3]/deta, dN[3]/dzeta
            dN[9] = (1.0 + xi[1]) * (1.0 - xi[2]) / (-8.0);
            dN[10] = (1.0 - xi[0]) * (1.0 - xi[2]) / (8.0);
            dN[11] = (1.0 - xi[0]) * (1.0 + xi[1]) / (-8.0);

            // dN[4]/dxi, dN[4]/deta, dN[4]/dzeta
            dN[12] = (1.0 - xi[1]) * (1.0 + xi[2]) / (-8.0);
            dN[13] = (1.0 - xi[0]) * (1.0 + xi[2]) / (-8.0);
            dN[14] = (1.0 - xi[0]) * (1.0 - xi[1]) / (8.0);

            // dN[5]/dxi, dN[5]/deta, dN[5]/dzeta
            dN[15] = (1.0 - xi[1]) * (1.0 + xi[2]) / (8.0);
            dN[16] = (1.0 + xi[0]) * (1.0 + xi[2]) / (-8.0);
            dN[17] = (1.0 + xi[0]) * (1.0 - xi[1]) / (8.0);

            // dN[6]/dxi, dN[6]/deta, dN[6]/dzeta
            dN[18] = (1.0 + xi[1]) * (1.0 + xi[2]) / (8.0);
            dN[19] = (1.0 + xi[0]) * (1.0 + xi[2]) / (8.0);
            dN[20] = (1.0 + xi[0]) * (1.0 + xi[1]) / (8.0);

            // dN[7]/dxi, dN[7]/deta, dN[7]/dzeta
            dN[21] = (1.0 + xi[1]) * (1.0 + xi[2]) / (-8.0);
            dN[22] = (1.0 - xi[0]) * (1.0 + xi[2]) / (8.0);
            dN[23] = (1.0 - xi[0]) * (1.0 + xi[1]) / (8.0);

        } else if (elemType == ELEMTYPE::TET4) {
            // dN[0]/dxi, dN[0]/deta, dN[0]/dzeta:
            dN[0] = -1.0;
            dN[1] = -1.0;
            dN[2] = -1.0;
            // dN[1]/dxi, dN[1]/deta, dN[1]/dzeta:
            dN[3] = 1.0;
            dN[4] = 0.0;
            dN[5] = 0.0;
            // dN[2]/dxi, dN[2]/deta, dN[2]/dzeta:
            dN[6] = 0.0;
            dN[7] = 1.0;
            dN[8] = 0.0;
            // dN[3]/dxi, dN[3]/deta, dN[3]/dzeta:
            dN[9] = 0.0;
            dN[10] = 0.0;
            dN[11] = 1.0;
        } else if (elemType == ELEMTYPE::TET10) {
            const DT L0 = 1.0 - xi[0] - xi[1] - xi[2];
            const DT L1 = xi[0];
            const DT L2 = xi[1];
            const DT L3 = xi[2];

            const DT dL0dxi = -1.0;
            const DT dL0det = -1.0;
            const DT dL0dze = -1.0;

            const DT dL1dxi = 1.0;
            const DT dL1det = 0.0;
            const DT dL1dze = 0.0;
            
            const DT dL2dxi = 0.0;
            const DT dL2det = 1.0;
            const DT dL2dze = 0.0;
            
            const DT dL3dxi = 0.0;
            const DT dL3det = 0.0;
            const DT dL3dze = 1.0;

            // dN[0]/dxi, dN[0]/deta, dN[0]/dzeta
            dN[0] = (4 * L0 * dL0dxi) - dL0dxi;
            dN[1] = (4 * L0 * dL0det) - dL0det;
            dN[2] = (4 * L0 * dL0dze) - dL0dze;
            // dN[1]/dxi, dN[1]/deta, dN[1]/dzeta
            dN[3] = (4 * L1 * dL1dxi) - dL1dxi;
            dN[4] = (4 * L1 * dL1det) - dL1det;
            dN[5] = (4 * L1 * dL1dze) - dL1dze;
            // dN[2]/dxi, dN[2]/deta, dN[2]/dzeta
            dN[6] = (4 * L2 * dL2dxi) - dL2dxi;
            dN[7] = (4 * L2 * dL2det) - dL2det;
            dN[8] = (4 * L2 * dL2dze) - dL2dze;
            // dN[3]/dxi, dN[3]/deta, dN[3]/dzeta
            dN[9] = (4 * L3 * dL3dxi) - dL3dxi;
            dN[10] = (4 * L3 * dL3det) - dL3det;
            dN[11] = (4 * L3 * dL3dze) - dL3dze;

            // dN[4]/dxi, dN[4]/deta, dN[4]/dzeta
            dN[12] = 4.0*(dL0dxi * L1 + L0 * dL1dxi);
            dN[13] = 4.0*(dL0det * L1 + L0 * dL1det);
            dN[14] = 4.0*(dL0dze * L1 + L0 * dL1dze);
            // dN[5]/dxi, dN[5]/deta, dN[5]/dzeta
            dN[15] = 4.0*(dL1dxi * L2 + L1 * dL2dxi);
            dN[16] = 4.0*(dL1det * L2 + L1 * dL2det);
            dN[17] = 4.0*(dL1dze * L2 + L1 * dL2dze);
            // dN[6]/dxi, dN[6]/deta, dN[6]/dzeta
            dN[18] = 4.0*(dL2dxi * L0 + L2 * dL0dxi);
            dN[19] = 4.0*(dL2det * L0 + L2 * dL0det);
            dN[20] = 4.0*(dL2dze * L0 + L2 * dL0dze);

            // dN[7]/dxi, dN[7]/deta, dN[7]/dzeta
            dN[21] = 4.0*(dL0dxi * L3 + L0 * dL3dxi);
            dN[22] = 4.0*(dL0det * L3 + L0 * dL3det);
            dN[23] = 4.0*(dL0dze * L3 + L0 * dL3dze);
            // dN[8]/dxi, dN[8]/deta, dN[8]/dzeta
            dN[24] = 4.0*(dL1dxi * L3 + L1 * dL3dxi);
            dN[25] = 4.0*(dL1det * L3 + L1 * dL3det);
            dN[26] = 4.0*(dL1dze * L3 + L1 * dL3dze);
            // dN[9]/dxi, dN[9]/deta, dN[9]/dzeta
            dN[27] = 4.0*(dL2dxi * L3 + L2 * dL3dxi);
            dN[28] = 4.0*(dL2det * L3 + L2 * dL3det);
            dN[29] = 4.0*(dL2dze * L3 + L2 * dL3dze);

        } else if (elemType == ELEMTYPE::QUAD4) {
            // linear 4-node quad element
            // dN[0]/dxi, dN[0]/deta
            dN[0] = 0.25 * (xi[1] - 1.0);
            dN[1] = 0.25 * (xi[0] - 1.0);

            // dN[1]/dxi, dN[1]/deta
            dN[2] = 0.25 * (1.0 - xi[1]);
            dN[3] = (-0.25) * (1.0 + xi[0]);

            // dN[2]/dxi, dN[2]/deta
            dN[4] = 0.25 * (xi[1] + 1.0);
            dN[5] = 0.25 * (xi[0] + 1.0);

            // dN[3]/dxi, dN[3]/deta
            dN[6] = (-0.25) * (1.0 + xi[1]);
            dN[7] = 0.25 * (1.0 - xi[0]);

        } else if (elemType == ELEMTYPE::QUAD8){
            // dN[0]/dxi, dN[0]/deta
            dN[0] = 0.25 * (1 - xi[1]) * (2*xi[0] + xi[1]);
            dN[1] = 0.25 * (1 - xi[0]) * (2*xi[1] + xi[0]);

            // dN[1]/dxi, dN[1]/deta
            dN[2] = 0.25 * (xi[1] - 1) * (-2*xi[0] + xi[1]);
            dN[3] = (-0.25) * (1 + xi[0]) * (-2*xi[1] + xi[0]);

            // dN[2]/dxi, dN[2]/deta
            dN[4] = (0.25) * (1 + xi[1]) * (2*xi[0] + xi[1]);
            dN[5] = (0.25) * (1 + xi[0]) * (2*xi[1] + xi[0]);

            // dN[3]/dxi, dN[3]/deta
            dN[6] = (-0.25) * (1 + xi[1]) * (-2*xi[0] + xi[1]);
            dN[7] = (0.25) * (xi[0] - 1) * (-2*xi[1] + xi[0]);

            // dN[4]/dxi, dN[4]/deta
            dN[8] = xi[0] * (xi[1] - 1);
            dN[9] = 0.5 * (xi[0]*xi[0] - 1);

            // dN[5]/dxi, dN[5]/deta
            dN[10] = 0.5 * (1 - xi[1]*xi[1]);
            dN[11] = -xi[1] * (1 + xi[0]);

            // dN[6]/dxi, dN[6]/deta
            dN[12] = -xi[0] * (1 + xi[1]);
            dN[13] = 0.5 * (1 - xi[0]*xi[0]);

            // dN[7]/dxi, dN[7]/deta
            dN[14] = 0.5 * (xi[1]*xi[1] - 1);
            dN[15] = -xi[1] * (1 - xi[0]);

        } else if (elemType == ELEMTYPE::TRI3){
            // dN[0]/dxi, dN[0]/deta
            dN[0] = -1.0;
            dN[1] = -1.0;
            // dN[1]/dxi, dN[1]/deta
            dN[2] = 1.0;
            dN[3] = 0.0;
            // dN[2]/dxi, dN[2]/deta
            dN[4] = 0.0;
            dN[5] = 1.0;

        } else if (elemType == ELEMTYPE::TRI6){
            dN[0] = 1. - 4.*(1.0-xi[0]-xi[1]);
            dN[1] = dN[0];
            dN[2] = 4.*xi[0] - 1.;
            dN[3] = 0.0;
            dN[4] = 0.0;
            dN[5] = 4.*xi[1] - 1.;
            dN[6] = 4.*((1.0-xi[0]-xi[1])-xi[0]);
            dN[7] = -4.*xi[0];
            dN[8] = 4.*xi[1];
            dN[9] = 4.*xi[0];
            dN[10] = -4.*xi[1];
            dN[11] = 4.*((1.0-xi[0]-xi[1])-xi[1]);

        } else if (elemType == ELEMTYPE::HEX20){
            // 20-node quadratic serendipity element
            for (unsigned int j = 0; j < 8; j++){
                if (j == 0){
                    xij = -1; etaj = -1; zetaj = -1;
                } else if (j == 1){
                    xij = 1; etaj = -1; zetaj = -1;
                } else if (j == 2){
                    xij = 1; etaj = 1; zetaj = -1;
                } else if (j == 3){
                    xij = -1; etaj = 1; zetaj = -1;
                } else if (j == 4){
                    xij = -1; etaj = -1; zetaj = 1;
                } else if (j == 5){
                    xij = 1; etaj = -1; zetaj = 1;
                } else if (j == 6){
                    xij = 1; etaj = 1; zetaj = 1;
                } else if (j == 7){
                    xij = -1; etaj = 1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                dN[j*3]     = 0.125 * (1 + etaj*xi[1])*(1 + zetaj*xi[2])*(xij + 2*xij*xij*xi[0] + xij*etaj*xi[1] + xij*zetaj*xi[2] - 2*xij);
                dN[j*3 + 1] = 0.125 * (1 + xij*xi[0])*(1 + zetaj*xi[2])*(etaj + 2*etaj*etaj*xi[1] + etaj*xij*xi[0] + etaj*zetaj*xi[2] - 2*etaj);
                dN[j*3 + 2] = 0.125 * (1 + xij*xi[0])*(1 + etaj*xi[1])*(zetaj + 2*zetaj*zetaj*xi[2] + zetaj*xij*xi[0] + zetaj*etaj*xi[1] - 2*zetaj);
            }
            for (unsigned int j = 8; j < 15; j += 2){
                if (j == 8){
                    etaj = -1; zetaj = -1;
                } else if (j == 10){
                    etaj = 1; zetaj = -1;
                } else if (j == 12){
                    etaj = -1; zetaj = 1;
                } else if (j == 14){
                    etaj = 1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                dN[j*3]     = 0.25 * (-2*xi[0]) * (1 + etaj*xi[1]) * (1 + zetaj*xi[2]);
                dN[j*3 + 1] = 0.25 * (1 - xi[0]*xi[0]) * etaj * (1 + zetaj*xi[2]);
                dN[j*3 + 2] = 0.25 * (1 - xi[0]*xi[0]) * zetaj * (1 + etaj*xi[1]);
            }
            for (unsigned int j = 9; j < 16; j += 2){
                if (j == 9){
                    xij = 1; zetaj = -1;
                } else if (j == 11){
                    xij = -1; zetaj = -1;
                } else if (j == 13){
                    xij = 1; zetaj = 1;
                } else if (j == 15){
                    xij = -1; zetaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                dN[j*3]     = 0.25 * (1 - xi[1]*xi[1]) * xij * (1 + zetaj*xi[2]);
                dN[j*3 + 1] = 0.25 * (-2*xi[1]) * (1 + xij*xi[0]) * (1 + zetaj*xi[2]);
                dN[j*3 + 2] = 0.25 * (1 - xi[1]*xi[1]) * zetaj * (1 + xij*xi[0]);
            }
            for (unsigned int j = 16; j < 20; j++){
                if (j == 16){
                    xij = -1; etaj = -1;
                } else if (j == 17){
                    xij = 1; etaj = -1;
                } else if (j == 18){
                    xij = 1; etaj = 1;
                } else if (j == 19){
                    xij = -1; etaj = 1;
                } else {
                    std::cout << "Impossible case!\n";
                    exit(0);
                }
                dN[j*3]     = 0.25 * (1 - xi[2]*xi[2]) * xij * (1 + etaj*xi[1]);
                dN[j*3 + 1] = 0.25 * (1 - xi[2]*xi[2]) * etaj * (1 + xij*xi[0]);
                dN[j*3 + 2] = 0.25 * (-2*xi[2]) * (1 + xij*xi[0]) * (1 + etaj*xi[1]);
            }    
        } else if (elemType == ELEMTYPE::HEX27) {
            const DT L0xi = LQuad(xi[0], 0);
            const DT L1xi = LQuad(xi[0], 1);
            const DT L2xi = LQuad(xi[0], 2);

            const DT L0eta = LQuad(xi[1], 0);
            const DT L1eta = LQuad(xi[1], 1);
            const DT L2eta = LQuad(xi[1], 2);

            const DT L0zeta = LQuad(xi[2], 0);
            const DT L1zeta = LQuad(xi[2], 1);
            const DT L2zeta = LQuad(xi[2], 2);
            
            const DT dL0xi = DLQuad(xi[0], 0); // dL0(xi)/dxi
            const DT dL1xi = DLQuad(xi[0], 1); // dL1(xi)/dxi
            const DT dL2xi = DLQuad(xi[0], 2); // dL2(xi)/dxi

            const DT dL0eta = DLQuad(xi[1], 0); // dL0(eta)/deta
            const DT dL1eta = DLQuad(xi[1], 1); // dL1(eta)/deta
            const DT dL2eta = DLQuad(xi[1], 2); // dL2(eta)/deta

            const DT dL0zeta = DLQuad(xi[2], 0); // dL0(zeta)/dzeta
            const DT dL1zeta = DLQuad(xi[2], 1); // dL1(zeta)/dzeta
            const DT dL2zeta = DLQuad(xi[2], 2); // dL2(zeta)/dzeta

            dN[0] = dL0xi * L0eta * L0zeta;
            dN[1] = L0xi * dL0eta * L0zeta;
            dN[2] = L0xi * L0eta * dL0zeta;

            dN[3] = dL2xi * L0eta * L0zeta;
            dN[4] = L2xi * dL0eta * L0zeta;
            dN[5] = L2xi * L0eta * dL0zeta;

            dN[6] = dL2xi * L2eta * L0zeta;
            dN[7] = L2xi * dL2eta * L0zeta;
            dN[8] = L2xi * L2eta * dL0zeta;

            dN[9] = dL0xi * L2eta * L0zeta;
            dN[10] = L0xi * dL2eta * L0zeta;
            dN[11] = L0xi * L2eta * dL0zeta;

            dN[12] = dL0xi * L0eta * L2zeta;
            dN[13] = L0xi * dL0eta * L2zeta;
            dN[14] = L0xi * L0eta * dL2zeta;

            dN[15] = dL2xi * L0eta * L2zeta;
            dN[16] = L2xi * dL0eta * L2zeta;
            dN[17] = L2xi * L0eta * dL2zeta;

            dN[18] = dL2xi * L2eta * L2zeta;
            dN[19] = L2xi * dL2eta * L2zeta;
            dN[20] = L2xi * L2eta * dL2zeta;

            dN[21] = dL0xi * L2eta * L2zeta;
            dN[22] = L0xi * dL2eta * L2zeta;
            dN[23] = L0xi * L2eta * dL2zeta;

            // 12 mid-side edge nodes:
            dN[24] = dL1xi * L0eta * L0zeta;
            dN[25] = L1xi * dL0eta * L0zeta;
            dN[26] = L1xi * L0eta * dL0zeta;

            dN[27] = dL0xi * L1eta * L0zeta;
            dN[28] = L0xi * dL1eta * L0zeta;
            dN[29] = L0xi * L1eta * dL0zeta;

            dN[30] = dL0xi * L0eta * L1zeta;
            dN[31] = L0xi * dL0eta * L1zeta;
            dN[32] = L0xi * L0eta * dL1zeta;

            dN[33] = dL2xi * L1eta * L0zeta;
            dN[34] = L2xi * dL1eta * L0zeta;
            dN[35] = L2xi * L1eta * dL0zeta;

            dN[36] = dL2xi * L0eta * L1zeta;
            dN[37] = L2xi * dL0eta * L1zeta;
            dN[38] = L2xi * L0eta * dL1zeta;

            dN[39] = dL1xi * L2eta * L0zeta;
            dN[40] = L1xi * dL2eta * L0zeta;
            dN[41] = L1xi * L2eta * dL0zeta;

            dN[42] = dL2xi * L2eta * L1zeta;
            dN[43] = L2xi * dL2eta * L1zeta;
            dN[44] = L2xi * L2eta * dL1zeta;

            dN[45] = dL0xi * L2eta * L1zeta;
            dN[46] = L0xi * dL2eta * L1zeta;
            dN[47] = L0xi * L2eta * dL1zeta;

            dN[48] = dL1xi * L0eta * L2zeta;
            dN[49] = L1xi * dL0eta * L2zeta;
            dN[50] = L1xi * L0eta * dL2zeta;

            dN[51] = dL0xi * L1eta * L2zeta;
            dN[52] = L0xi * dL1eta * L2zeta;
            dN[53] = L0xi * L1eta * dL2zeta;

            dN[54] = dL2xi * L1eta * L2zeta;
            dN[55] = L2xi * dL1eta * L2zeta;
            dN[56] = L2xi * L1eta * dL2zeta;

            dN[57] = dL1xi * L2eta * L2zeta;
            dN[58] = L1xi * dL2eta * L2zeta;
            dN[59] = L1xi * L2eta * dL2zeta;

            // 6 face-center nodes
            dN[60] = dL1xi * L1eta * L0zeta;
            dN[61] = L1xi * dL1eta * L0zeta;
            dN[62] = L1xi * L1eta * dL0zeta;

            dN[63] = dL1xi * L0eta * L1zeta;
            dN[64] = L1xi * dL0eta * L1zeta;
            dN[65] = L1xi * L0eta * dL1zeta;

            dN[66] = dL0xi * L1eta * L1zeta;
            dN[67] = L0xi * dL1eta * L1zeta;
            dN[68] = L0xi * L1eta * dL1zeta;

            dN[69] = dL2xi * L1eta * L1zeta;
            dN[70] = L2xi * dL1eta * L1zeta;
            dN[71] = L2xi * L1eta * dL1zeta;

            dN[72] = dL1xi * L2eta * L1zeta;
            dN[73] = L1xi * dL2eta * L1zeta;
            dN[74] = L1xi * L2eta * dL1zeta;

            dN[75] = dL1xi * L1eta * L2zeta;
            dN[76] = L1xi * dL1eta * L2zeta;
            dN[77] = L1xi * L1eta * dL2zeta;

            // 1 center node
            dN[78] = dL1xi * L1eta * L1zeta;
            dN[79] = L1xi * dL1eta * L1zeta;
            dN[80] = L1xi * L1eta * dL1zeta;
            
        } else {
            std::cout << "Not implemented yet\n";
            exit(0);
        }
        return dN;
    }
}// namespace shape

#endif //ADAPTIVEMATRIX_SHAPEFUNC_H
