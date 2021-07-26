package com.nvidia.triton.contrib;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class UtilTest {


    @Test
    void testIsEmptyString() {
        assertTrue(Util.isEmpty((String) null));
        assertTrue(Util.isEmpty(""));
    }

    @Test
    void testIsEmptyCollection() {
        assertTrue(Util.isEmpty((List<String>)null));
        assertTrue(Util.isEmpty(Collections.emptyList()));
    }

    @Test
    void testElemNumFromShape() {
        assertEquals(Util.elemNumFromShape(new long[] {1, 2, 3}), 6L);
    }

    @Test
    void testIntToBytes() {
        byte[] bytes = Util.intToBytes(23423910);
        assertEquals(Arrays.toString(bytes), "[-90, 107, 101, 1]");
    }
}