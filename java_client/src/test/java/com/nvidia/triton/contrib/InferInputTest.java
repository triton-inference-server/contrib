package com.nvidia.triton.contrib;

import java.lang.reflect.Method;
import java.util.stream.Stream;

import com.google.common.io.BaseEncoding;
import com.google.common.primitives.UnsignedInteger;
import com.google.common.primitives.UnsignedLong;
import com.nvidia.triton.contrib.pojo.DataType;
import com.nvidia.triton.contrib.pojo.IOTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author xiafei.qiuxf
 * @date 2021/4/19
 */
public class InferInputTest {

    private static Stream<Arguments> testData() {
        // (dataType,dataArray,expectedJSON, expectedBytes)
        // expectedBytes are in string format for readability and they're produced from triton's python SDK.
        return Stream.of(
            Arguments.of(DataType.BOOL, new boolean[] {true, false, true, false, true, false},
                "[true,false,true,false,true,false]",
                "010001000100"),
            Arguments.of(DataType.INT8, new byte[] {1, 2, 3, 4, -5, -6},
                "[1,2,3,4,-5,-6]",
                "01020304fbfa"),
            Arguments.of(DataType.INT16, new short[] {1, 2, 3, 4, -5, -6},
                "[1,2,3,4,-5,-6]",
                "0100020003000400fbfffaff"),
            Arguments.of(DataType.INT32, new int[] {1, 2, 3, 4, -5, -6},
                "[1,2,3,4,-5,-6]",
                "01000000020000000300000004000000fbfffffffaffffff"),
            Arguments.of(DataType.UINT8, new byte[] {1, 2, 3, 4, 5, 6},
                "[1,2,3,4,5,6]",
                "010203040506"),
            Arguments.of(DataType.UINT16, new short[] {1, 2, 3, 4, 5, 6},
                "[1,2,3,4,5,6]",
                "010002000300040005000600"),
            Arguments.of(DataType.UINT32, new int[] {1, 2, 3, 4,
                    UnsignedInteger.valueOf("4294967200").intValue(),
                    UnsignedInteger.valueOf("4294967201").intValue(),
                },
                "[1,2,3,4,4294967200,4294967201]",
                "01000000020000000300000004000000a0ffffffa1ffffff"),
            Arguments.of(DataType.INT64, new long[] {1, 2, 3, 4, -5, -6},
                "[1,2,3,4,-5,-6]",
                "0100000000000000020000000000000003000000000000000400000000000000fbfffffffffffffffaffffffffffffff"),
            Arguments.of(DataType.UINT64, new long[] {1, 2, 3, 4,
                    UnsignedLong.valueOf("10446744073709551615").longValue(),
                    UnsignedLong.valueOf("12446744073709551615").longValue()
                },
                "[1,2,3,4,10446744073709551615,12446744073709551615]",
                "0100000000000000020000000000000003000000000000000400000000000000ffffdfc4624afa90ffffa713cab7bbac"),
            Arguments.of(DataType.FP32, new float[] {1.1F, 2.2F, 3.3F, 4.4F, 5.5F, 6.6F},
                "[1.1,2.2,3.3,4.4,5.5,6.6]",
                "cdcc8c3fcdcc0c4033335340cdcc8c400000b0403333d340"),
            Arguments.of(DataType.FP64, new double[] {1.1, 2.2, 3.3, 4.4, 5.5, 6.6},
                "[1.1,2.2,3.3,4.4,5.5,6.6]",
                "9a9999999999f13f9a999999999901406666666666660a409a9999999999114000000000000016406666666666661a40"),
            Arguments.of(DataType.BYTES, new String[] {"aa", "bb", "cc", "dd", "ee", "ff"},
                "[\"aa\",\"bb\",\"cc\",\"dd\",\"ee\",\"ff\"]",
                "020000006161020000006262020000006363020000006464020000006565020000006666")
        );
    }

    @ParameterizedTest
    @MethodSource("testData")
    public void testJSON_NormalCases(DataType dataType, Object data, String expectedJSON) throws Exception {
        // Create an InferInput object and call setData via reflection.
        long[] shape = {3, 2};
        InferInput input = new InferInput("foo", shape, dataType);
        Method setData = InferInput.class.getDeclaredMethod("setData", data.getClass(), boolean.class);
        setData.invoke(input, data, false);

        // Check data as expected after serialize to JSON.
        assertNull(input.getBinaryData());
        String jsonStr = Util.toJson(input.getJSONData());
        assertEquals(expectedJSON, jsonStr);

        // Check IOTensor.
        IOTensor tensor = input.getTensor();
        assertNull(tensor.getParameters().getInt("binary_data_size"));
        assertEquals(tensor.getName(), "foo");
        assertEquals(tensor.getDatatype(), dataType);
        assertArrayEquals(tensor.getShape(), shape);
        assertNotNull(tensor.getData());
    }

    @ParameterizedTest
    @MethodSource("testData")
    public void testBinary_NormalCases(DataType dataType, Object data, String expectedJSON, String expectedBytes)
        throws Exception {
        // Create an InferInput object and call setData via reflection.
        long[] shape = {3, 2};
        InferInput input = new InferInput("foo", shape, dataType);
        Method setData = InferInput.class.getDeclaredMethod("setData", data.getClass(), boolean.class);
        setData.invoke(input, data, true);

        // Check data as expected after serialize to bytes.
        assertNull(input.getJSONData());
        String hexData = BaseEncoding.base16().lowerCase().encode(input.getBinaryData());
        assertEquals(expectedBytes, hexData);

        // Check IOTensor.
        IOTensor tensor = input.getTensor();
        Integer binSize = tensor.getParameters().getInt("binary_data_size");
        assertNotNull(binSize);
        assertTrue(binSize > 0);
        assertEquals(tensor.getName(), "foo");
        assertEquals(tensor.getDatatype(), dataType);
        assertArrayEquals(tensor.getShape(), shape);
        assertNull(tensor.getData());
    }

    @Test
    void testDataNotSet() {
        InferInput input = new InferInput("foo", new long[] {3, 2}, DataType.FP32);
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, input::getTensor);
        assertEquals(ex.getMessage(), ".setData method not call on InferInput foo");

        input = new InferInput("foo2", new long[] {3, 2}, DataType.INT32);
        input.setData(new int[] {1, 1, 1, 1, 1, 1}, true);
        input.getTensor();

        input = new InferInput("foo3", new long[] {3, 2}, DataType.INT32);
        input.setData(new int[] {1, 1, 1, 1, 1, 1}, false);
        input.getTensor();
    }
}
