package com.nvidia.triton.contrib;

import com.nvidia.triton.contrib.pojo.IOTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/20
 */
class InferRequestedOutputTest {
    @Test
    void testGetTensor() {
        InferRequestedOutput out = new InferRequestedOutput("out1", false, 3);
        IOTensor tensor = out.getTensor();
        assertEquals("out1", tensor.getName());
        assertFalse(tensor.getParameters().getBool("binary_data"));
        assertEquals(3, tensor.getParameters().getInt("classification"));
    }
}