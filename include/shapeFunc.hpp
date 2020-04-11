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

    enum class ELEMTYPE {QUAD4, QUAD8, HEX8, HEX20};
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
                case shape::ELEMTYPE::QUAD4: return 4; break;
                case shape::ELEMTYPE::QUAD8: return 8; break;
                case shape::ELEMTYPE::HEX8: return 8; break;
                case shape::ELEMTYPE::HEX20: return 20; break;
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
    };

    template <typename DT>
    shapeFunc<DT>::shapeFunc(ELEMTYPE eltype){
        
        elemType = eltype;
        
        numNodes = nNodesPerElem(eltype);

        // allocate memory for N and dN
        if (numNodes > 0){
            N = new DT [numNodes];
            if ((elemType == shape::ELEMTYPE::QUAD4) || 
                (elemType == shape::ELEMTYPE::QUAD8)){
                dN = new DT [ 2 * numNodes ];
            } else if ((elemType == shape::ELEMTYPE::HEX8) || 
                    (elemType == shape::ELEMTYPE::HEX20)){
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
        } else if (elemType == ELEMTYPE::QUAD4){
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
        
        } else {
            std::cout << "Not implemented yet\n";
            exit(0);
        }
        return dN;
    }
}// namespace shape

#endif //ADAPTIVEMATRIX_SHAPEFUNC_H
