// Convert the network's zero-based class index (0..104) to the original
// subject number (1..109). Subjects S088, S092, S100 and S104 are absent
// from the trained 105-class dataset.
module class_to_subject_id (
    input  logic [6:0] class_index,
    output logic [6:0] subject_id
);
    always_comb begin
        subject_id = class_index + 7'd1;

        if (class_index >= 7'd87)
            subject_id = subject_id + 7'd1;
        if (class_index >= 7'd90)
            subject_id = subject_id + 7'd1;
        if (class_index >= 7'd97)
            subject_id = subject_id + 7'd1;
        if (class_index >= 7'd100)
            subject_id = subject_id + 7'd1;
    end
endmodule
