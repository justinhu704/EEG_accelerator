`timescale 1ns/1ps

module tb_class_to_subject_id;
    logic [6:0] class_index;
    logic [6:0] subject_id;
    integer errors;

    class_to_subject_id dut (
        .class_index(class_index),
        .subject_id(subject_id)
    );

    task automatic check_mapping(
        input logic [6:0] test_class,
        input logic [6:0] expected_subject
    );
        begin
            class_index = test_class;
            #1;
            if (subject_id !== expected_subject) begin
                $display("FAIL: class %0d mapped to S%03d, expected S%03d",
                         test_class, subject_id, expected_subject);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        errors = 0;

        check_mapping(7'd0,   7'd1);    // S001
        check_mapping(7'd19,  7'd20);   // S020
        check_mapping(7'd86,  7'd87);   // immediately before skipped S088
        check_mapping(7'd87,  7'd89);   // immediately after skipped S088
        check_mapping(7'd89,  7'd91);   // immediately before skipped S092
        check_mapping(7'd90,  7'd93);   // immediately after skipped S092
        check_mapping(7'd96,  7'd99);   // immediately before skipped S100
        check_mapping(7'd97,  7'd101);  // immediately after skipped S100
        check_mapping(7'd99,  7'd103);  // immediately before skipped S104
        check_mapping(7'd100, 7'd105);  // immediately after skipped S104
        check_mapping(7'd104, 7'd109);  // final class

        if (errors == 0)
            $display("PASS: all class indices map to the expected subject IDs.");
        else
            $fatal(1, "Subject mapping failed with %0d errors.", errors);
        $finish;
    end
endmodule
