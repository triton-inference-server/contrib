package com.nvidia.triton.contrib.pojo;

import com.nvidia.triton.contrib.Util;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author xiafei.qiuxf
 * @date 2021/7/26
 */
class DataTypeTest {
    @Test
    void testJson() throws Exception {
        final String s = Util.toJson(DataType.BOOL);
        assertEquals("\"BOOL\"", s);
        DataType dType = Util.fromJson(s, DataType.class);
        assertEquals(DataType.BOOL, dType);
    }
}