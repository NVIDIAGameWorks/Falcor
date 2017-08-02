#include "CommonShaderDescs.h"

namespace Falcor {

    namespace CommonShaderDescs
    {

        //  Return the Shader Code Block.
        std::string getCodeBlock(const std::vector<std::string> &lines)
        {
            std::string codeBlock = "";

            //
            for (uint32_t avIndex = 0; avIndex < lines.size(); avIndex++)
            {
                codeBlock = codeBlock + lines[avIndex] + " \n";
            }

            return codeBlock;
        }

    };
};
