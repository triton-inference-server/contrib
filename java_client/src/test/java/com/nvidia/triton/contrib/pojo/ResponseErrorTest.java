package com.nvidia.triton.contrib.pojo;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.nvidia.triton.contrib.Util;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author xiafei.qiuxf
 * @date 2021/7/28
 */
class ResponseErrorTest {

    @Test
    void testJson() throws JsonProcessingException {
        ResponseError err = new ResponseError();
        err.setError("hi");
        String s = Util.toJson(err);
        ResponseError err2 = Util.fromJson(s, ResponseError.class);
        assertEquals("hi", err2.getError());
    }
}