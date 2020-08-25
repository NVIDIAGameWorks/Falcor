/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/** Slots on the interior list have the following bit layout.
    0-26  materialID
    27    active bit
    28-31 nestedPriority

    We put the nestedPriority in the highest value bits in order to simplify sorting the list.
*/

#ifndef INTERIOR_LIST_SLOT_COUNT
#define INTERIOR_LIST_SLOT_COUNT 2
#endif

struct InteriorList
{
    static const uint kNoMaterial           = 0xffffffff;

    static const uint kMaterialBits         = 27;
    static const uint kActiveBits           = 1;
    static const uint kNestedPriorityBits   = 4;

    static const uint kMaterialOffset       = 0;
    static const uint kActiveOffset         = kMaterialOffset + kMaterialBits;
    static const uint kNestedPriorityOffset = kActiveOffset + kActiveBits;

    static const uint kMaterialMask         = ((1 << kMaterialBits) - 1) << kMaterialOffset;
    static const uint kActiveMask           = ((1 << kActiveBits) - 1) << kActiveOffset;
    static const uint kNestedPriorityMask   = ((1 << kNestedPriorityBits) - 1) << kNestedPriorityOffset;

    uint slots[INTERIOR_LIST_SLOT_COUNT];

    /** Make an active material slot given the material.
        \param[in] materialID Material ID.
        \param[in] nestedPriority Nested priority.
        \return Returns the encoded slot.
    */
    uint makeSlot(uint materialID, uint nestedPriority)
    {
        // Get nested priority of given material.
        return (nestedPriority << kNestedPriorityOffset) | kActiveMask | (materialID & kMaterialMask);
    }

    /** Check if a slot is active.
        \param[in] slot Slot value.
        \return Returns true if slot is active.
    */
    bool isSlotActive(uint slot)
    {
        return slot & kActiveMask;
    }

    /** Check if the interior list is empty.
        \return Returns true if interior list is empty, false otherwise.
    */
    bool isEmpty()
    {
        return !isSlotActive(slots[0]);
    }

    /** Get the nested priority from a slot.
        \param[in] slot Slot value.
        \param Returns the nested priority or 0 for empty slots.
    */
    uint getSlotNestedPriority(uint slot)
    {
        return slot >> kNestedPriorityOffset;
    }

    /** Get the material ID from a slot.
        \param[in] slot Slot value.
        \return Returns the material ID or 0 for empty slots.
    */
    uint getSlotMaterialID(uint slot)
    {
        return slot & kMaterialMask;
    }

    /** Return current highest nested priority on interior list.
        Because the interior list is sorted by nested priority and empty slots have priority 0,
        we can simple fetch the first slot.
        \return Returns the highest nested priority on the interior list.
    */
    uint getTopNestedPriority()
    {
        return getSlotNestedPriority(slots[0]);
    }

    /** Return the material ID with the highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getTopMaterialID()
    {
        return isSlotActive(slots[0]) ? getSlotMaterialID(slots[0]) : kNoMaterial;
    }

    /** Return the material ID with the 2nd highest priority on the interior list.
        \return Returns the material ID or kNoMaterial if interior list is empty.
    */
    uint getNextMaterialID()
    {
        return isSlotActive(slots[1]) ? getSlotMaterialID(slots[1]) : kNoMaterial;
    }

    /** Check if an intersection with a given surface is a true intersection.
        True intersection occurs if nested priority of intersected mesh is equal or higher
        priority than the highest nested priority on the interior list.
        \param[in] nestedPriority Nested priority of intersected surface.
        \return Returns true if intersection is a true intersection, false otherwise.
    */
    bool isTrueIntersection(uint nestedPriority)
    {
        // Get nested priority of given material.
        return nestedPriority >= getTopNestedPriority();
    }

    /** Handle an intersection with a given material:
        If material is already on interior list -> remove it.
        If material is not on interior list -> add it.
        \param[in] materialID Material ID of intersected material.
        \param[in] nestedPriority Nested priority of intersected surface.
        \param[in] entering True if material is entered, false if material is left.
    */
    [mutating] void handleIntersection(uint materialID, uint nestedPriority, bool entering)
    {
        for (uint slotIndex = 0; slotIndex < INTERIOR_LIST_SLOT_COUNT; ++slotIndex)
        {
            uint slot = slots[slotIndex];

            // Add new slot to interior list
            if (entering && slot == 0)
            {
                slots[slotIndex] = makeSlot(materialID, nestedPriority);
                break;
            }

            // Remove existing slot from interior list
            if (!entering && isSlotActive(slot) && getSlotMaterialID(slot) == materialID)
            {
                slots[slotIndex] = 0;
                break;
            }
        }

        sortSlots();
    }

    /** Sort the interior list by priority.
    */
    [mutating] void sortSlots()
    {
#define CSWAP(_a, _b)               \
        if (slots[_a] < slots[_b])  \
        {                           \
            uint tmp = slots[_a];   \
            slots[_a] = slots[_b];  \
            slots[_b] = tmp;        \
        }

        // sorting networks: http://pages.ripco.net/~jgamble/nw.html
#if INTERIOR_LIST_SLOT_COUNT == 2
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 3
        CSWAP(0, 1)
        CSWAP(1, 2)
        CSWAP(0, 1)
#elif INTERIOR_LIST_SLOT_COUNT == 4
        CSWAP(0, 1)
        CSWAP(2, 3)
        CSWAP(0, 2)
        CSWAP(1, 3)
        CSWAP(1, 2)
#else
#error "sorting not handled for given number of slots"
#endif

#undef CSWAP
    }
};
