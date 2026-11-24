struct type_6_ {
    member: array<f32>,
}

struct type_10_ {
    member: array<vec4<f32>>,
}

struct type_19_ {
    member: u32,
    member_1_: u32,
}

struct type_25_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: u32,
    member_6_: u32,
    member_7_: u32,
}

struct type_26_ {
    member: type_25_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> sg_lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_26_;
@group(0) @binding(1) 
var<storage, read_write> block_inputs_or_d_loss_d_inputs: type_10_;
@group(0) @binding(2) 
var<storage> tile_descriptors: type_10_;
@group(0) @binding(3) 
var<storage> feature_outputs: type_10_;
@group(0) @binding(4) 
var<storage, read_write> tile_weight_grads: type_6_;
@group(0) @binding(5) 
var<storage, read_write> d_loss_d_features: type_10_;
var<workgroup> reduce_a: array<f32, 256>;
var<workgroup> reduce_b: array<f32, 256>;

fn function_() {
    var local: array<array<u32, 4>, 4>;
    var local_1_: array<array<u32, 4>, 4>;
    var phi_4185_: bool;
    var phi_3817_: type_19_;
    var phi_3818_: type_19_;
    var phi_3841_: type_19_;
    var phi_4183_: bool;
    var phi_3852_: u32;
    var phi_3855_: f32;
    var phi_3857_: f32;
    var phi_3866_: u32;
    var phi_3869_: f32;
    var phi_3871_: f32;
    var phi_3896_: u32;
    var phi_3916_: u32;
    var phi_3867_: u32;
    var phi_3870_: f32;
    var phi_3872_: f32;
    var phi_4180_: bool;
    var phi_4193_: bool;
    var phi_3853_: u32;
    var phi_3856_: f32;
    var phi_3858_: f32;
    var phi_4192_: bool;
    var local_2_: f32;
    var phi_3937_: bool;
    var local_3_: f32;
    var phi_3953_: bool;
    var phi_3991_: u32;
    var phi_3992_: u32;
    var phi_4194_: bool;
    var phi_4052_: bool;
    var phi_4070_: bool;
    var phi_4090_: bool;
    var phi_4107_: u32;
    var phi_4124_: bool;
    var phi_4141_: u32;
    var phi_4206_: bool;
    var phi_4205_: bool;
    var phi_4428_: bool;
    var phi_2304_: vec4<f32>;
    var phi_2344_: vec4<f32>;
    var phi_2357_: vec4<f32>;
    var phi_2370_: vec4<f32>;
    var phi_2383_: vec4<f32>;
    var phi_2396_: vec4<f32>;
    var phi_2409_: vec4<f32>;
    var phi_2421_: vec4<f32>;
    var phi_2434_: vec4<f32>;
    var phi_2443_: f32;
    var phi_2444_: f32;
    var phi_2445_: f32;
    var phi_2446_: f32;
    var phi_2447_: f32;
    var phi_2448_: f32;
    var phi_2449_: f32;
    var phi_2450_: f32;
    var phi_2451_: vec4<f32>;
    var phi_2452_: vec4<f32>;
    var phi_2453_: vec4<f32>;
    var phi_2454_: vec4<f32>;
    var phi_2455_: vec4<f32>;
    var phi_2456_: vec4<f32>;
    var phi_2458_: f32;
    var phi_2459_: f32;
    var phi_2460_: f32;
    var phi_2461_: f32;
    var phi_2462_: f32;
    var phi_2463_: f32;
    var phi_2464_: f32;
    var phi_2465_: f32;
    var phi_2466_: vec4<f32>;
    var phi_2467_: vec4<f32>;
    var phi_2468_: vec4<f32>;
    var phi_2469_: vec4<f32>;
    var phi_2470_: vec4<f32>;
    var phi_2471_: vec4<f32>;
    var phi_2473_: bool;
    var phi_2474_: f32;
    var phi_2475_: f32;
    var phi_2476_: f32;
    var phi_2477_: f32;
    var phi_2478_: f32;
    var phi_2479_: f32;
    var phi_2480_: f32;
    var phi_2481_: f32;
    var phi_2482_: vec4<f32>;
    var phi_2483_: vec4<f32>;
    var phi_2484_: vec4<f32>;
    var phi_2485_: vec4<f32>;
    var phi_2486_: vec4<f32>;
    var phi_2487_: vec4<f32>;
    var phi_2488_: bool;
    var phi_2489_: bool;
    var phi_4233_: bool;
    var phi_2527_: type_19_;
    var phi_2528_: type_19_;
    var phi_2551_: type_19_;
    var phi_2569_: f32;
    var phi_2580_: f32;
    var phi_2597_: bool;
    var phi_2613_: bool;
    var phi_2651_: u32;
    var phi_2652_: u32;
    var phi_4220_: bool;
    var phi_2712_: bool;
    var phi_2730_: bool;
    var phi_2752_: bool;
    var phi_2768_: bool;
    var phi_2806_: u32;
    var phi_2807_: u32;
    var phi_4251_: bool;
    var phi_2867_: bool;
    var phi_2885_: bool;
    var phi_2901_: f32;
    var phi_2912_: f32;
    var phi_2929_: bool;
    var phi_2945_: bool;
    var phi_2983_: u32;
    var phi_2984_: u32;
    var phi_4266_: bool;
    var phi_3044_: bool;
    var phi_3062_: bool;
    var phi_3084_: bool;
    var phi_3100_: bool;
    var phi_3138_: u32;
    var phi_3139_: u32;
    var phi_4283_: bool;
    var phi_3199_: bool;
    var phi_3217_: bool;
    var phi_3233_: f32;
    var phi_3244_: f32;
    var phi_3261_: bool;
    var phi_3277_: bool;
    var phi_3315_: u32;
    var phi_3316_: u32;
    var phi_4298_: bool;
    var phi_3376_: bool;
    var phi_3394_: bool;
    var phi_3416_: bool;
    var phi_3432_: bool;
    var phi_3470_: u32;
    var phi_3471_: u32;
    var phi_4315_: bool;
    var phi_3531_: bool;
    var phi_3549_: bool;
    var phi_3571_: u32;
    var phi_3590_: u32;
    var phi_3609_: u32;
    var phi_3628_: u32;
    var phi_3647_: u32;
    var phi_3667_: u32;
    var phi_3687_: u32;
    var phi_3707_: u32;
    var phi_3726_: u32;
    var phi_3746_: u32;
    var phi_3766_: u32;
    var phi_3786_: u32;
    var phi_4331_: bool;
    var phi_4330_: bool;
    var phi_4430_: bool;
    var phi_4429_: bool;
    var phi_1046_: u32;
    var phi_1047_: u32;
    var phi_1048_: u32;
    var phi_1077_: vec4<f32>;
    var phi_1090_: vec4<f32>;
    var phi_1103_: vec4<f32>;
    var phi_1116_: vec4<f32>;
    var phi_1129_: vec4<f32>;
    var phi_1142_: vec4<f32>;
    var phi_1145_: type_19_;
    var phi_1148_: vec4<f32>;
    var phi_1150_: vec4<f32>;
    var phi_1152_: vec4<f32>;
    var phi_1154_: vec4<f32>;
    var phi_1156_: vec4<f32>;
    var phi_1158_: vec4<f32>;
    var phi_1146_: type_19_;
    var phi_1181_: type_19_;
    var phi_1201_: vec4<f32>;
    var phi_1230_: vec4<f32>;
    var phi_1243_: vec4<f32>;
    var phi_1256_: vec4<f32>;
    var phi_1269_: vec4<f32>;
    var phi_1282_: vec4<f32>;
    var phi_1295_: vec4<f32>;
    var phi_1308_: vec4<f32>;
    var phi_1321_: vec4<f32>;
    var phi_1334_: vec4<f32>;
    var phi_1347_: vec4<f32>;
    var phi_1360_: vec4<f32>;
    var phi_1373_: vec4<f32>;
    var phi_1386_: vec4<f32>;
    var phi_1399_: vec4<f32>;
    var phi_1772_: vec4<f32>;
    var phi_1773_: vec4<f32>;
    var phi_1774_: vec4<f32>;
    var phi_1775_: vec4<f32>;
    var phi_1776_: vec4<f32>;
    var phi_1777_: vec4<f32>;
    var phi_1778_: vec4<f32>;
    var phi_1779_: vec4<f32>;
    var phi_1780_: vec4<f32>;
    var phi_1781_: vec4<f32>;
    var phi_1782_: vec4<f32>;
    var phi_1783_: vec4<f32>;
    var phi_1149_: vec4<f32>;
    var phi_1151_: vec4<f32>;
    var phi_1153_: vec4<f32>;
    var phi_1155_: vec4<f32>;
    var phi_1157_: vec4<f32>;
    var phi_1159_: vec4<f32>;
    var phi_4361_: bool;
    var local_4_: vec4<f32>;
    var phi_1795_: bool;
    var local_5_: vec4<f32>;
    var phi_1812_: bool;
    var local_6_: vec4<f32>;
    var phi_1829_: bool;
    var local_7_: vec4<f32>;
    var phi_1846_: bool;
    var local_8_: vec4<f32>;
    var phi_1875_: bool;
    var local_9_: vec4<f32>;
    var phi_1892_: bool;
    var local_10_: vec4<f32>;
    var phi_1909_: bool;
    var local_11_: vec4<f32>;
    var phi_1926_: bool;
    var local_12_: vec4<f32>;
    var phi_1955_: bool;
    var local_13_: vec4<f32>;
    var phi_1972_: bool;
    var local_14_: vec4<f32>;
    var phi_1989_: bool;
    var local_15_: vec4<f32>;
    var phi_2006_: bool;
    var local_16_: vec4<f32>;
    var phi_2035_: bool;
    var local_17_: vec4<f32>;
    var phi_2052_: bool;
    var local_18_: vec4<f32>;
    var phi_2069_: bool;
    var local_19_: vec4<f32>;
    var phi_2086_: bool;
    var local_20_: vec4<f32>;
    var phi_2115_: bool;
    var local_21_: vec4<f32>;
    var phi_2132_: bool;
    var local_22_: vec4<f32>;
    var phi_2149_: bool;
    var local_23_: vec4<f32>;
    var phi_2166_: bool;
    var local_24_: vec4<f32>;
    var phi_2195_: bool;
    var local_25_: vec4<f32>;
    var phi_2212_: bool;
    var local_26_: vec4<f32>;
    var phi_2229_: bool;
    var local_27_: vec4<f32>;
    var phi_2246_: bool;
    var phi_4432_: bool;
    var phi_4431_: bool;
    var phi_158_: u32;
    var phi_159_: u32;
    var phi_188_: vec4<f32>;
    var phi_201_: vec4<f32>;
    var phi_204_: type_19_;
    var phi_207_: vec4<f32>;
    var phi_209_: vec4<f32>;
    var phi_205_: type_19_;
    var phi_232_: type_19_;
    var phi_252_: vec4<f32>;
    var phi_281_: vec4<f32>;
    var phi_294_: vec4<f32>;
    var phi_307_: vec4<f32>;
    var phi_320_: vec4<f32>;
    var phi_333_: vec4<f32>;
    var phi_346_: vec4<f32>;
    var phi_359_: vec4<f32>;
    var phi_372_: vec4<f32>;
    var phi_385_: vec4<f32>;
    var phi_398_: vec4<f32>;
    var phi_411_: vec4<f32>;
    var phi_424_: vec4<f32>;
    var phi_437_: vec4<f32>;
    var phi_450_: vec4<f32>;
    var phi_463_: vec4<f32>;
    var phi_476_: vec4<f32>;
    var phi_489_: vec4<f32>;
    var phi_502_: vec4<f32>;
    var phi_825_: vec4<f32>;
    var phi_826_: vec4<f32>;
    var phi_827_: vec4<f32>;
    var phi_828_: vec4<f32>;
    var phi_208_: vec4<f32>;
    var phi_210_: vec4<f32>;
    var phi_4395_: bool;
    var local_28_: vec4<f32>;
    var phi_840_: bool;
    var local_29_: vec4<f32>;
    var phi_857_: bool;
    var local_30_: vec4<f32>;
    var phi_874_: bool;
    var local_31_: vec4<f32>;
    var phi_891_: bool;
    var local_32_: vec4<f32>;
    var phi_920_: bool;
    var local_33_: vec4<f32>;
    var phi_937_: bool;
    var local_34_: vec4<f32>;
    var phi_954_: bool;
    var local_35_: vec4<f32>;
    var phi_971_: bool;
    var phi_4488_: bool;
    var phi_4487_: bool;
    var phi_4427_: bool;
    var local_36_: f32;
    var local_37_: f32;

    switch bitcast<i32>(0u) {
        default: {
            let _e55_ = wg_id_1_;
            let _e56_ = lid_1_;
            let _e57_ = sg_lid_1_;
            let _e59_ = arrayLength((&block_inputs_or_d_loss_d_inputs.member));
            let _e61_ = arrayLength((&tile_descriptors.member));
            let _e63_ = arrayLength((&feature_outputs.member));
            let _e65_ = arrayLength((&tile_weight_grads.member));
            let _e67_ = arrayLength((&d_loss_d_features.member));
            let _e68_ = (_e56_ & 255u);
            let _e71_ = uniforms.member.member_3_;
            switch bitcast<i32>(_e71_) {
                case 0: {
                    let _e1922_ = uniforms.member.member;
                    let _e1925_ = uniforms.member.member_6_;
                    let _e1927_ = select(1u, _e1925_, (1u < _e1925_));
                    let _e1930_ = uniforms.member.member_2_;
                    let _e1933_ = uniforms.member.member_5_;
                    if (_e1933_ == 4294967295u) {
                        phi_159_ = select(16u, _e55_.x, (_e55_.x < _e1927_));
                    } else {
                        let _e1936_ = (_e55_.x < 4u);
                        if _e1936_ {
                            local_1_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 6u, 8u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 7u, 16u, 16u), array<u32, 4>(3u, 16u, 16u, 16u));
                            let _e1937_ = (_e1933_ & 3u);
                            if (_e1937_ < 4u) {
                            } else {
                                phi_4427_ = true;
                                break;
                            }
                            if _e1936_ {
                            } else {
                                phi_4427_ = true;
                                break;
                            }
                            let _e1941_ = local_1_[_e1937_][_e55_.x];
                            phi_158_ = select(16u, _e1941_, (_e1941_ < _e1927_));
                        } else {
                            phi_158_ = 16u;
                        }
                        let _e1945_ = phi_158_;
                        phi_159_ = _e1945_;
                    }
                    let _e1949_ = phi_159_;
                    if (_e1949_ >= _e1927_) {
                        phi_4487_ = false;
                    } else {
                        let _e1953_ = ((_e55_.y * 256u) + _e68_);
                        if (_e1953_ >= _e1922_) {
                            phi_4488_ = false;
                        } else {
                            let _e1956_ = (_e55_.z * _e1927_);
                            let _e1960_ = ((((_e1956_ + _e1949_) * _e1922_) + _e1953_) * 2u);
                            let _e1961_ = (_e1960_ < _e59_);
                            if _e1961_ {
                                if _e1961_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1964_ = block_inputs_or_d_loss_d_inputs.member[_e1960_];
                                phi_188_ = _e1964_;
                            } else {
                                phi_188_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1966_ = phi_188_;
                            let _e1967_ = (_e1960_ + 1u);
                            let _e1968_ = (_e1967_ < _e59_);
                            if _e1968_ {
                                if _e1968_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1971_ = block_inputs_or_d_loss_d_inputs.member[_e1967_];
                                phi_201_ = _e1971_;
                            } else {
                                phi_201_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1973_ = phi_201_;
                            phi_204_ = type_19_(0u, _e1930_);
                            phi_207_ = _e1973_;
                            phi_209_ = _e1966_;
                            loop {
                                let _e1976_ = phi_204_;
                                let _e1978_ = phi_207_;
                                let _e1980_ = phi_209_;
                                local_28_ = _e1980_;
                                local_29_ = _e1980_;
                                local_30_ = _e1980_;
                                local_31_ = _e1980_;
                                local_32_ = _e1978_;
                                local_33_ = _e1978_;
                                local_34_ = _e1978_;
                                local_35_ = _e1978_;
                                if (_e1976_.member < _e1976_.member_1_) {
                                    phi_205_ = type_19_((_e1976_.member + 1u), _e1976_.member_1_);
                                    phi_232_ = type_19_(1u, _e1976_.member);
                                } else {
                                    phi_205_ = _e1976_;
                                    phi_232_ = type_19_(0u, u32());
                                }
                                let _e1993_ = phi_205_;
                                let _e1995_ = phi_232_;
                                let _e1999_ = (bitcast<i32>(_e1995_.member) != 0i);
                                if _e1999_ {
                                    let _e2000_ = (_e1995_.member_1_ * 13u);
                                    let _e2001_ = (_e2000_ < _e61_);
                                    if _e2001_ {
                                        if _e2001_ {
                                        } else {
                                            phi_4395_ = true;
                                            break;
                                        }
                                        let _e2004_ = tile_descriptors.member[_e2000_];
                                        phi_252_ = _e2004_;
                                    } else {
                                        phi_252_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e2006_ = phi_252_;
                                    let _e2008_ = bitcast<u32>(_e2006_.x);
                                    if (bitcast<u32>(_e2006_.y) == _e1949_) {
                                        if (_e2008_ < _e1927_) {
                                            let _e2016_ = ((((_e1956_ + _e2008_) * _e1922_) + _e1953_) * 6u);
                                            let _e2017_ = (_e2016_ < _e63_);
                                            if _e2017_ {
                                                if _e2017_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2020_ = feature_outputs.member[_e2016_];
                                                phi_281_ = _e2020_;
                                            } else {
                                                phi_281_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2022_ = phi_281_;
                                            let _e2023_ = (_e2016_ + 1u);
                                            let _e2024_ = (_e2023_ < _e63_);
                                            if _e2024_ {
                                                if _e2024_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2027_ = feature_outputs.member[_e2023_];
                                                phi_294_ = _e2027_;
                                            } else {
                                                phi_294_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2029_ = phi_294_;
                                            let _e2030_ = (_e2016_ + 2u);
                                            let _e2031_ = (_e2030_ < _e63_);
                                            if _e2031_ {
                                                if _e2031_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2034_ = feature_outputs.member[_e2030_];
                                                phi_307_ = _e2034_;
                                            } else {
                                                phi_307_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2036_ = phi_307_;
                                            let _e2037_ = (_e2016_ + 3u);
                                            let _e2038_ = (_e2037_ < _e63_);
                                            if _e2038_ {
                                                if _e2038_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2041_ = feature_outputs.member[_e2037_];
                                                phi_320_ = _e2041_;
                                            } else {
                                                phi_320_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2043_ = phi_320_;
                                            let _e2044_ = (_e2016_ + 4u);
                                            let _e2045_ = (_e2044_ < _e63_);
                                            if _e2045_ {
                                                if _e2045_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2048_ = feature_outputs.member[_e2044_];
                                                phi_333_ = _e2048_;
                                            } else {
                                                phi_333_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2050_ = phi_333_;
                                            let _e2051_ = (_e2016_ + 5u);
                                            let _e2052_ = (_e2051_ < _e63_);
                                            if _e2052_ {
                                                if _e2052_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2055_ = feature_outputs.member[_e2051_];
                                                phi_346_ = _e2055_;
                                            } else {
                                                phi_346_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2057_ = phi_346_;
                                            let _e2058_ = (_e2000_ + 1u);
                                            let _e2059_ = (_e2058_ < _e61_);
                                            if _e2059_ {
                                                if _e2059_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2062_ = tile_descriptors.member[_e2058_];
                                                phi_359_ = _e2062_;
                                            } else {
                                                phi_359_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2064_ = phi_359_;
                                            let _e2065_ = (_e2000_ + 2u);
                                            let _e2066_ = (_e2065_ < _e61_);
                                            if _e2066_ {
                                                if _e2066_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2069_ = tile_descriptors.member[_e2065_];
                                                phi_372_ = _e2069_;
                                            } else {
                                                phi_372_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2071_ = phi_372_;
                                            let _e2072_ = (_e2000_ + 3u);
                                            let _e2073_ = (_e2072_ < _e61_);
                                            if _e2073_ {
                                                if _e2073_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2076_ = tile_descriptors.member[_e2072_];
                                                phi_385_ = _e2076_;
                                            } else {
                                                phi_385_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2078_ = phi_385_;
                                            let _e2079_ = (_e2000_ + 4u);
                                            let _e2080_ = (_e2079_ < _e61_);
                                            if _e2080_ {
                                                if _e2080_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2083_ = tile_descriptors.member[_e2079_];
                                                phi_398_ = _e2083_;
                                            } else {
                                                phi_398_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2085_ = phi_398_;
                                            let _e2086_ = (_e2000_ + 5u);
                                            let _e2087_ = (_e2086_ < _e61_);
                                            if _e2087_ {
                                                if _e2087_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2090_ = tile_descriptors.member[_e2086_];
                                                phi_411_ = _e2090_;
                                            } else {
                                                phi_411_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2092_ = phi_411_;
                                            let _e2093_ = (_e2000_ + 6u);
                                            let _e2094_ = (_e2093_ < _e61_);
                                            if _e2094_ {
                                                if _e2094_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2097_ = tile_descriptors.member[_e2093_];
                                                phi_424_ = _e2097_;
                                            } else {
                                                phi_424_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2099_ = phi_424_;
                                            let _e2100_ = (_e2000_ + 7u);
                                            let _e2101_ = (_e2100_ < _e61_);
                                            if _e2101_ {
                                                if _e2101_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2104_ = tile_descriptors.member[_e2100_];
                                                phi_437_ = _e2104_;
                                            } else {
                                                phi_437_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2106_ = phi_437_;
                                            let _e2107_ = (_e2000_ + 8u);
                                            let _e2108_ = (_e2107_ < _e61_);
                                            if _e2108_ {
                                                if _e2108_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2111_ = tile_descriptors.member[_e2107_];
                                                phi_450_ = _e2111_;
                                            } else {
                                                phi_450_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2113_ = phi_450_;
                                            let _e2114_ = (_e2000_ + 9u);
                                            let _e2115_ = (_e2114_ < _e61_);
                                            if _e2115_ {
                                                if _e2115_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2118_ = tile_descriptors.member[_e2114_];
                                                phi_463_ = _e2118_;
                                            } else {
                                                phi_463_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2120_ = phi_463_;
                                            let _e2121_ = (_e2000_ + 10u);
                                            let _e2122_ = (_e2121_ < _e61_);
                                            if _e2122_ {
                                                if _e2122_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2125_ = tile_descriptors.member[_e2121_];
                                                phi_476_ = _e2125_;
                                            } else {
                                                phi_476_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2127_ = phi_476_;
                                            let _e2128_ = (_e2000_ + 11u);
                                            let _e2129_ = (_e2128_ < _e61_);
                                            if _e2129_ {
                                                if _e2129_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2132_ = tile_descriptors.member[_e2128_];
                                                phi_489_ = _e2132_;
                                            } else {
                                                phi_489_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2134_ = phi_489_;
                                            let _e2135_ = (_e2000_ + 12u);
                                            let _e2136_ = (_e2135_ < _e61_);
                                            if _e2136_ {
                                                if _e2136_ {
                                                } else {
                                                    phi_4395_ = true;
                                                    break;
                                                }
                                                let _e2139_ = tile_descriptors.member[_e2135_];
                                                phi_502_ = _e2139_;
                                            } else {
                                                phi_502_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2141_ = phi_502_;
                                            phi_825_ = vec4<f32>(((((((((((((_e1978_.x + (_e2064_.x * _e2043_.x)) + (_e2071_.x * _e2043_.y)) + (_e2078_.x * _e2043_.z)) + (_e2085_.x * _e2043_.w)) + (_e2092_.x * _e2057_.x)) + (_e2099_.x * _e2057_.y)) + (_e2106_.x * _e2057_.z)) + (_e2113_.x * _e2057_.w)) + (_e2120_.x * _e2029_.x)) + (_e2127_.x * _e2029_.y)) + (_e2134_.x * _e2029_.z)) + (_e2141_.x * _e2029_.w)), ((((((((((((_e1978_.y + (_e2064_.y * _e2043_.x)) + (_e2071_.y * _e2043_.y)) + (_e2078_.y * _e2043_.z)) + (_e2085_.y * _e2043_.w)) + (_e2092_.y * _e2057_.x)) + (_e2099_.y * _e2057_.y)) + (_e2106_.y * _e2057_.z)) + (_e2113_.y * _e2057_.w)) + (_e2120_.y * _e2029_.x)) + (_e2127_.y * _e2029_.y)) + (_e2134_.y * _e2029_.z)) + (_e2141_.y * _e2029_.w)), ((((((((((((_e1978_.z + (_e2064_.z * _e2043_.x)) + (_e2071_.z * _e2043_.y)) + (_e2078_.z * _e2043_.z)) + (_e2085_.z * _e2043_.w)) + (_e2092_.z * _e2057_.x)) + (_e2099_.z * _e2057_.y)) + (_e2106_.z * _e2057_.z)) + (_e2113_.z * _e2057_.w)) + (_e2120_.z * _e2029_.x)) + (_e2127_.z * _e2029_.y)) + (_e2134_.z * _e2029_.z)) + (_e2141_.z * _e2029_.w)), ((((((((((((_e1978_.w + (_e2064_.w * _e2043_.x)) + (_e2071_.w * _e2043_.y)) + (_e2078_.w * _e2043_.z)) + (_e2085_.w * _e2043_.w)) + (_e2092_.w * _e2057_.x)) + (_e2099_.w * _e2057_.y)) + (_e2106_.w * _e2057_.z)) + (_e2113_.w * _e2057_.w)) + (_e2120_.w * _e2029_.x)) + (_e2127_.w * _e2029_.y)) + (_e2134_.w * _e2029_.z)) + (_e2141_.w * _e2029_.w)));
                                            phi_826_ = vec4<f32>(((((((((((((_e1980_.x + (_e2064_.x * _e2036_.x)) + (_e2071_.x * _e2036_.y)) + (_e2078_.x * _e2036_.z)) + (_e2085_.x * _e2036_.w)) + (_e2092_.x * _e2050_.x)) + (_e2099_.x * _e2050_.y)) + (_e2106_.x * _e2050_.z)) + (_e2113_.x * _e2050_.w)) + (_e2120_.x * _e2022_.x)) + (_e2127_.x * _e2022_.y)) + (_e2134_.x * _e2022_.z)) + (_e2141_.x * _e2022_.w)), ((((((((((((_e1980_.y + (_e2064_.y * _e2036_.x)) + (_e2071_.y * _e2036_.y)) + (_e2078_.y * _e2036_.z)) + (_e2085_.y * _e2036_.w)) + (_e2092_.y * _e2050_.x)) + (_e2099_.y * _e2050_.y)) + (_e2106_.y * _e2050_.z)) + (_e2113_.y * _e2050_.w)) + (_e2120_.y * _e2022_.x)) + (_e2127_.y * _e2022_.y)) + (_e2134_.y * _e2022_.z)) + (_e2141_.y * _e2022_.w)), ((((((((((((_e1980_.z + (_e2064_.z * _e2036_.x)) + (_e2071_.z * _e2036_.y)) + (_e2078_.z * _e2036_.z)) + (_e2085_.z * _e2036_.w)) + (_e2092_.z * _e2050_.x)) + (_e2099_.z * _e2050_.y)) + (_e2106_.z * _e2050_.z)) + (_e2113_.z * _e2050_.w)) + (_e2120_.z * _e2022_.x)) + (_e2127_.z * _e2022_.y)) + (_e2134_.z * _e2022_.z)) + (_e2141_.z * _e2022_.w)), ((((((((((((_e1980_.w + (_e2064_.w * _e2036_.x)) + (_e2071_.w * _e2036_.y)) + (_e2078_.w * _e2036_.z)) + (_e2085_.w * _e2036_.w)) + (_e2092_.w * _e2050_.x)) + (_e2099_.w * _e2050_.y)) + (_e2106_.w * _e2050_.z)) + (_e2113_.w * _e2050_.w)) + (_e2120_.w * _e2022_.x)) + (_e2127_.w * _e2022_.y)) + (_e2134_.w * _e2022_.z)) + (_e2141_.w * _e2022_.w)));
                                        } else {
                                            phi_825_ = _e1978_;
                                            phi_826_ = _e1980_;
                                        }
                                        let _e2417_ = phi_825_;
                                        let _e2419_ = phi_826_;
                                        phi_827_ = _e2417_;
                                        phi_828_ = _e2419_;
                                    } else {
                                        phi_827_ = _e1978_;
                                        phi_828_ = _e1980_;
                                    }
                                    let _e2421_ = phi_827_;
                                    let _e2423_ = phi_828_;
                                    phi_208_ = _e2421_;
                                    phi_210_ = _e2423_;
                                } else {
                                    phi_208_ = vec4<f32>();
                                    phi_210_ = vec4<f32>();
                                }
                                let _e2425_ = phi_208_;
                                let _e2427_ = phi_210_;
                                phi_204_ = _e1993_;
                                phi_207_ = _e2425_;
                                phi_209_ = _e2427_;
                                phi_4395_ = false;
                                if !(_e1999_) {
                                    break;
                                }
                            }
                            let _e2430_ = phi_4395_;
                            phi_4427_ = _e2430_;
                            if _e2430_ {
                                break;
                            }
                            let _e2432_ = local_28_;
                            let _e2434_ = bitcast<u32>(_e2432_.x);
                            let _e2435_ = (_e2434_ & 2139095040u);
                            if (_e2435_ == 2139095040u) {
                                phi_840_ = true;
                            } else {
                                phi_840_ = (_e2435_ == 0u);
                            }
                            let _e2439_ = phi_840_;
                            let _e2443_ = local_29_;
                            let _e2445_ = bitcast<u32>(_e2443_.y);
                            let _e2446_ = (_e2445_ & 2139095040u);
                            if (_e2446_ == 2139095040u) {
                                phi_857_ = true;
                            } else {
                                phi_857_ = (_e2446_ == 0u);
                            }
                            let _e2450_ = phi_857_;
                            let _e2454_ = local_30_;
                            let _e2456_ = bitcast<u32>(_e2454_.z);
                            let _e2457_ = (_e2456_ & 2139095040u);
                            if (_e2457_ == 2139095040u) {
                                phi_874_ = true;
                            } else {
                                phi_874_ = (_e2457_ == 0u);
                            }
                            let _e2461_ = phi_874_;
                            let _e2465_ = local_31_;
                            let _e2467_ = bitcast<u32>(_e2465_.w);
                            let _e2468_ = (_e2467_ & 2139095040u);
                            if (_e2468_ == 2139095040u) {
                                phi_891_ = true;
                            } else {
                                phi_891_ = (_e2468_ == 0u);
                            }
                            let _e2472_ = phi_891_;
                            if _e1961_ {
                                if _e1961_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                block_inputs_or_d_loss_d_inputs.member[_e1960_] = vec4<f32>(bitcast<f32>(select(_e2434_, 0u, _e2439_)), bitcast<f32>(select(_e2445_, 0u, _e2450_)), bitcast<f32>(select(_e2456_, 0u, _e2461_)), bitcast<f32>(select(_e2467_, 0u, _e2472_)));
                            }
                            let _e2479_ = local_32_;
                            let _e2481_ = bitcast<u32>(_e2479_.x);
                            let _e2482_ = (_e2481_ & 2139095040u);
                            if (_e2482_ == 2139095040u) {
                                phi_920_ = true;
                            } else {
                                phi_920_ = (_e2482_ == 0u);
                            }
                            let _e2486_ = phi_920_;
                            let _e2490_ = local_33_;
                            let _e2492_ = bitcast<u32>(_e2490_.y);
                            let _e2493_ = (_e2492_ & 2139095040u);
                            if (_e2493_ == 2139095040u) {
                                phi_937_ = true;
                            } else {
                                phi_937_ = (_e2493_ == 0u);
                            }
                            let _e2497_ = phi_937_;
                            let _e2501_ = local_34_;
                            let _e2503_ = bitcast<u32>(_e2501_.z);
                            let _e2504_ = (_e2503_ & 2139095040u);
                            if (_e2504_ == 2139095040u) {
                                phi_954_ = true;
                            } else {
                                phi_954_ = (_e2504_ == 0u);
                            }
                            let _e2508_ = phi_954_;
                            let _e2512_ = local_35_;
                            let _e2514_ = bitcast<u32>(_e2512_.w);
                            let _e2515_ = (_e2514_ & 2139095040u);
                            if (_e2515_ == 2139095040u) {
                                phi_971_ = true;
                            } else {
                                phi_971_ = (_e2515_ == 0u);
                            }
                            let _e2519_ = phi_971_;
                            if _e1968_ {
                                if _e1968_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                block_inputs_or_d_loss_d_inputs.member[_e1967_] = vec4<f32>(bitcast<f32>(select(_e2481_, 0u, _e2486_)), bitcast<f32>(select(_e2492_, 0u, _e2497_)), bitcast<f32>(select(_e2503_, 0u, _e2508_)), bitcast<f32>(select(_e2514_, 0u, _e2519_)));
                            }
                            phi_4488_ = _e2430_;
                        }
                        let _e2526_ = phi_4488_;
                        phi_4487_ = _e2526_;
                    }
                    let _e2528_ = phi_4487_;
                    phi_4427_ = _e2528_;
                    break;
                }
                case 1: {
                    let _e1126_ = uniforms.member.member;
                    let _e1129_ = uniforms.member.member_6_;
                    let _e1131_ = select(1u, _e1129_, (1u < _e1129_));
                    let _e1134_ = uniforms.member.member_2_;
                    let _e1137_ = uniforms.member.member_5_;
                    let _e1138_ = (_e1137_ == 4294967295u);
                    if _e1138_ {
                        phi_1048_ = select(4294967295u, _e55_.x, (_e55_.x < _e1131_));
                    } else {
                        if _e1138_ {
                            phi_1047_ = select(16u, _e55_.x, (_e55_.x < _e1131_));
                        } else {
                            let _e1140_ = (_e55_.x < 4u);
                            if _e1140_ {
                                local = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 6u, 8u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 7u, 16u, 16u), array<u32, 4>(3u, 16u, 16u, 16u));
                                let _e1141_ = (_e1137_ & 3u);
                                if (_e1141_ < 4u) {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                if _e1140_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1145_ = local[_e1141_][_e55_.x];
                                phi_1046_ = select(16u, _e1145_, (_e1145_ < _e1131_));
                            } else {
                                phi_1046_ = 16u;
                            }
                            let _e1149_ = phi_1046_;
                            phi_1047_ = _e1149_;
                        }
                        let _e1153_ = phi_1047_;
                        phi_1048_ = _e1153_;
                    }
                    let _e1158_ = phi_1048_;
                    if (_e1158_ >= _e1131_) {
                        phi_4431_ = false;
                    } else {
                        let _e1162_ = ((_e55_.y * 256u) + _e68_);
                        if (_e1162_ >= _e1126_) {
                            phi_4432_ = false;
                        } else {
                            let _e1165_ = (_e55_.z * _e1131_);
                            let _e1169_ = ((((_e1165_ + _e1158_) * _e1126_) + _e1162_) * 6u);
                            let _e1170_ = (_e1169_ < _e67_);
                            if _e1170_ {
                                if _e1170_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1173_ = d_loss_d_features.member[_e1169_];
                                phi_1077_ = _e1173_;
                            } else {
                                phi_1077_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1175_ = phi_1077_;
                            let _e1176_ = (_e1169_ + 1u);
                            let _e1177_ = (_e1176_ < _e67_);
                            if _e1177_ {
                                if _e1177_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1180_ = d_loss_d_features.member[_e1176_];
                                phi_1090_ = _e1180_;
                            } else {
                                phi_1090_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1182_ = phi_1090_;
                            let _e1183_ = (_e1169_ + 2u);
                            let _e1184_ = (_e1183_ < _e67_);
                            if _e1184_ {
                                if _e1184_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1187_ = d_loss_d_features.member[_e1183_];
                                phi_1103_ = _e1187_;
                            } else {
                                phi_1103_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1189_ = phi_1103_;
                            let _e1190_ = (_e1169_ + 3u);
                            let _e1191_ = (_e1190_ < _e67_);
                            if _e1191_ {
                                if _e1191_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1194_ = d_loss_d_features.member[_e1190_];
                                phi_1116_ = _e1194_;
                            } else {
                                phi_1116_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1196_ = phi_1116_;
                            let _e1197_ = (_e1169_ + 4u);
                            let _e1198_ = (_e1197_ < _e67_);
                            if _e1198_ {
                                if _e1198_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1201_ = d_loss_d_features.member[_e1197_];
                                phi_1129_ = _e1201_;
                            } else {
                                phi_1129_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1203_ = phi_1129_;
                            let _e1204_ = (_e1169_ + 5u);
                            let _e1205_ = (_e1204_ < _e67_);
                            if _e1205_ {
                                if _e1205_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e1208_ = d_loss_d_features.member[_e1204_];
                                phi_1142_ = _e1208_;
                            } else {
                                phi_1142_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1210_ = phi_1142_;
                            phi_1145_ = type_19_(0u, _e1134_);
                            phi_1148_ = _e1210_;
                            phi_1150_ = _e1203_;
                            phi_1152_ = _e1196_;
                            phi_1154_ = _e1189_;
                            phi_1156_ = _e1182_;
                            phi_1158_ = _e1175_;
                            loop {
                                let _e1213_ = phi_1145_;
                                let _e1215_ = phi_1148_;
                                let _e1217_ = phi_1150_;
                                let _e1219_ = phi_1152_;
                                let _e1221_ = phi_1154_;
                                let _e1223_ = phi_1156_;
                                let _e1225_ = phi_1158_;
                                local_4_ = _e1225_;
                                local_5_ = _e1225_;
                                local_6_ = _e1225_;
                                local_7_ = _e1225_;
                                local_8_ = _e1223_;
                                local_9_ = _e1223_;
                                local_10_ = _e1223_;
                                local_11_ = _e1223_;
                                local_12_ = _e1221_;
                                local_13_ = _e1221_;
                                local_14_ = _e1221_;
                                local_15_ = _e1221_;
                                local_16_ = _e1219_;
                                local_17_ = _e1219_;
                                local_18_ = _e1219_;
                                local_19_ = _e1219_;
                                local_20_ = _e1217_;
                                local_21_ = _e1217_;
                                local_22_ = _e1217_;
                                local_23_ = _e1217_;
                                local_24_ = _e1215_;
                                local_25_ = _e1215_;
                                local_26_ = _e1215_;
                                local_27_ = _e1215_;
                                if (_e1213_.member < _e1213_.member_1_) {
                                    phi_1146_ = type_19_((_e1213_.member + 1u), _e1213_.member_1_);
                                    phi_1181_ = type_19_(1u, _e1213_.member);
                                } else {
                                    phi_1146_ = _e1213_;
                                    phi_1181_ = type_19_(0u, u32());
                                }
                                let _e1238_ = phi_1146_;
                                let _e1240_ = phi_1181_;
                                let _e1244_ = (bitcast<i32>(_e1240_.member) != 0i);
                                if _e1244_ {
                                    let _e1245_ = (_e1240_.member_1_ * 13u);
                                    let _e1246_ = (_e1245_ < _e61_);
                                    if _e1246_ {
                                        if _e1246_ {
                                        } else {
                                            phi_4361_ = true;
                                            break;
                                        }
                                        let _e1249_ = tile_descriptors.member[_e1245_];
                                        phi_1201_ = _e1249_;
                                    } else {
                                        phi_1201_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1251_ = phi_1201_;
                                    let _e1255_ = bitcast<u32>(_e1251_.y);
                                    if (bitcast<u32>(_e1251_.x) == _e1158_) {
                                        if (_e1255_ < _e1131_) {
                                            let _e1261_ = ((((_e1165_ + _e1255_) * _e1126_) + _e1162_) * 2u);
                                            let _e1262_ = (_e1261_ < _e59_);
                                            if _e1262_ {
                                                if _e1262_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1265_ = block_inputs_or_d_loss_d_inputs.member[_e1261_];
                                                phi_1230_ = _e1265_;
                                            } else {
                                                phi_1230_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1267_ = phi_1230_;
                                            let _e1268_ = (_e1261_ + 1u);
                                            let _e1269_ = (_e1268_ < _e59_);
                                            if _e1269_ {
                                                if _e1269_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1272_ = block_inputs_or_d_loss_d_inputs.member[_e1268_];
                                                phi_1243_ = _e1272_;
                                            } else {
                                                phi_1243_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1274_ = phi_1243_;
                                            let _e1275_ = (_e1245_ + 1u);
                                            let _e1276_ = (_e1275_ < _e61_);
                                            if _e1276_ {
                                                if _e1276_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1279_ = tile_descriptors.member[_e1275_];
                                                phi_1256_ = _e1279_;
                                            } else {
                                                phi_1256_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1281_ = phi_1256_;
                                            let _e1282_ = (_e1245_ + 2u);
                                            let _e1283_ = (_e1282_ < _e61_);
                                            if _e1283_ {
                                                if _e1283_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1286_ = tile_descriptors.member[_e1282_];
                                                phi_1269_ = _e1286_;
                                            } else {
                                                phi_1269_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1288_ = phi_1269_;
                                            let _e1289_ = (_e1245_ + 3u);
                                            let _e1290_ = (_e1289_ < _e61_);
                                            if _e1290_ {
                                                if _e1290_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1293_ = tile_descriptors.member[_e1289_];
                                                phi_1282_ = _e1293_;
                                            } else {
                                                phi_1282_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1295_ = phi_1282_;
                                            let _e1296_ = (_e1245_ + 4u);
                                            let _e1297_ = (_e1296_ < _e61_);
                                            if _e1297_ {
                                                if _e1297_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1300_ = tile_descriptors.member[_e1296_];
                                                phi_1295_ = _e1300_;
                                            } else {
                                                phi_1295_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1302_ = phi_1295_;
                                            let _e1303_ = (_e1245_ + 5u);
                                            let _e1304_ = (_e1303_ < _e61_);
                                            if _e1304_ {
                                                if _e1304_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1307_ = tile_descriptors.member[_e1303_];
                                                phi_1308_ = _e1307_;
                                            } else {
                                                phi_1308_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1309_ = phi_1308_;
                                            let _e1310_ = (_e1245_ + 6u);
                                            let _e1311_ = (_e1310_ < _e61_);
                                            if _e1311_ {
                                                if _e1311_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1314_ = tile_descriptors.member[_e1310_];
                                                phi_1321_ = _e1314_;
                                            } else {
                                                phi_1321_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1316_ = phi_1321_;
                                            let _e1317_ = (_e1245_ + 7u);
                                            let _e1318_ = (_e1317_ < _e61_);
                                            if _e1318_ {
                                                if _e1318_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1321_ = tile_descriptors.member[_e1317_];
                                                phi_1334_ = _e1321_;
                                            } else {
                                                phi_1334_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1323_ = phi_1334_;
                                            let _e1324_ = (_e1245_ + 8u);
                                            let _e1325_ = (_e1324_ < _e61_);
                                            if _e1325_ {
                                                if _e1325_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1328_ = tile_descriptors.member[_e1324_];
                                                phi_1347_ = _e1328_;
                                            } else {
                                                phi_1347_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1330_ = phi_1347_;
                                            let _e1331_ = (_e1245_ + 9u);
                                            let _e1332_ = (_e1331_ < _e61_);
                                            if _e1332_ {
                                                if _e1332_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1335_ = tile_descriptors.member[_e1331_];
                                                phi_1360_ = _e1335_;
                                            } else {
                                                phi_1360_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1337_ = phi_1360_;
                                            let _e1338_ = (_e1245_ + 10u);
                                            let _e1339_ = (_e1338_ < _e61_);
                                            if _e1339_ {
                                                if _e1339_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1342_ = tile_descriptors.member[_e1338_];
                                                phi_1373_ = _e1342_;
                                            } else {
                                                phi_1373_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1344_ = phi_1373_;
                                            let _e1345_ = (_e1245_ + 11u);
                                            let _e1346_ = (_e1345_ < _e61_);
                                            if _e1346_ {
                                                if _e1346_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1349_ = tile_descriptors.member[_e1345_];
                                                phi_1386_ = _e1349_;
                                            } else {
                                                phi_1386_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1351_ = phi_1386_;
                                            let _e1352_ = (_e1245_ + 12u);
                                            let _e1353_ = (_e1352_ < _e61_);
                                            if _e1353_ {
                                                if _e1353_ {
                                                } else {
                                                    phi_4361_ = true;
                                                    break;
                                                }
                                                let _e1356_ = tile_descriptors.member[_e1352_];
                                                phi_1399_ = _e1356_;
                                            } else {
                                                phi_1399_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1358_ = phi_1399_;
                                            phi_1772_ = (_e1215_ + vec4<f32>(((((_e1309_.x * _e1274_.x) + (_e1309_.y * _e1274_.y)) + (_e1309_.z * _e1274_.z)) + (_e1309_.w * _e1274_.w)), ((((_e1316_.x * _e1274_.x) + (_e1316_.y * _e1274_.y)) + (_e1316_.z * _e1274_.z)) + (_e1316_.w * _e1274_.w)), ((((_e1323_.x * _e1274_.x) + (_e1323_.y * _e1274_.y)) + (_e1323_.z * _e1274_.z)) + (_e1323_.w * _e1274_.w)), ((((_e1330_.x * _e1274_.x) + (_e1330_.y * _e1274_.y)) + (_e1330_.z * _e1274_.z)) + (_e1330_.w * _e1274_.w))));
                                            phi_1773_ = (_e1217_ + vec4<f32>(((((_e1309_.x * _e1267_.x) + (_e1309_.y * _e1267_.y)) + (_e1309_.z * _e1267_.z)) + (_e1309_.w * _e1267_.w)), ((((_e1316_.x * _e1267_.x) + (_e1316_.y * _e1267_.y)) + (_e1316_.z * _e1267_.z)) + (_e1316_.w * _e1267_.w)), ((((_e1323_.x * _e1267_.x) + (_e1323_.y * _e1267_.y)) + (_e1323_.z * _e1267_.z)) + (_e1323_.w * _e1267_.w)), ((((_e1330_.x * _e1267_.x) + (_e1330_.y * _e1267_.y)) + (_e1330_.z * _e1267_.z)) + (_e1330_.w * _e1267_.w))));
                                            phi_1774_ = (_e1219_ + vec4<f32>(((((_e1281_.x * _e1274_.x) + (_e1281_.y * _e1274_.y)) + (_e1281_.z * _e1274_.z)) + (_e1281_.w * _e1274_.w)), ((((_e1288_.x * _e1274_.x) + (_e1288_.y * _e1274_.y)) + (_e1288_.z * _e1274_.z)) + (_e1288_.w * _e1274_.w)), ((((_e1295_.x * _e1274_.x) + (_e1295_.y * _e1274_.y)) + (_e1295_.z * _e1274_.z)) + (_e1295_.w * _e1274_.w)), ((((_e1302_.x * _e1274_.x) + (_e1302_.y * _e1274_.y)) + (_e1302_.z * _e1274_.z)) + (_e1302_.w * _e1274_.w))));
                                            phi_1775_ = (_e1221_ + vec4<f32>(((((_e1281_.x * _e1267_.x) + (_e1281_.y * _e1267_.y)) + (_e1281_.z * _e1267_.z)) + (_e1281_.w * _e1267_.w)), ((((_e1288_.x * _e1267_.x) + (_e1288_.y * _e1267_.y)) + (_e1288_.z * _e1267_.z)) + (_e1288_.w * _e1267_.w)), ((((_e1295_.x * _e1267_.x) + (_e1295_.y * _e1267_.y)) + (_e1295_.z * _e1267_.z)) + (_e1295_.w * _e1267_.w)), ((((_e1302_.x * _e1267_.x) + (_e1302_.y * _e1267_.y)) + (_e1302_.z * _e1267_.z)) + (_e1302_.w * _e1267_.w))));
                                            phi_1776_ = (_e1223_ + vec4<f32>(((((_e1337_.x * _e1274_.x) + (_e1337_.y * _e1274_.y)) + (_e1337_.z * _e1274_.z)) + (_e1337_.w * _e1274_.w)), ((((_e1344_.x * _e1274_.x) + (_e1344_.y * _e1274_.y)) + (_e1344_.z * _e1274_.z)) + (_e1344_.w * _e1274_.w)), ((((_e1351_.x * _e1274_.x) + (_e1351_.y * _e1274_.y)) + (_e1351_.z * _e1274_.z)) + (_e1351_.w * _e1274_.w)), ((((_e1358_.x * _e1274_.x) + (_e1358_.y * _e1274_.y)) + (_e1358_.z * _e1274_.z)) + (_e1358_.w * _e1274_.w))));
                                            phi_1777_ = (_e1225_ + vec4<f32>(((((_e1337_.x * _e1267_.x) + (_e1337_.y * _e1267_.y)) + (_e1337_.z * _e1267_.z)) + (_e1337_.w * _e1267_.w)), ((((_e1344_.x * _e1267_.x) + (_e1344_.y * _e1267_.y)) + (_e1344_.z * _e1267_.z)) + (_e1344_.w * _e1267_.w)), ((((_e1351_.x * _e1267_.x) + (_e1351_.y * _e1267_.y)) + (_e1351_.z * _e1267_.z)) + (_e1351_.w * _e1267_.w)), ((((_e1358_.x * _e1267_.x) + (_e1358_.y * _e1267_.y)) + (_e1358_.z * _e1267_.z)) + (_e1358_.w * _e1267_.w))));
                                        } else {
                                            phi_1772_ = _e1215_;
                                            phi_1773_ = _e1217_;
                                            phi_1774_ = _e1219_;
                                            phi_1775_ = _e1221_;
                                            phi_1776_ = _e1223_;
                                            phi_1777_ = _e1225_;
                                        }
                                        let _e1596_ = phi_1772_;
                                        let _e1598_ = phi_1773_;
                                        let _e1600_ = phi_1774_;
                                        let _e1602_ = phi_1775_;
                                        let _e1604_ = phi_1776_;
                                        let _e1606_ = phi_1777_;
                                        phi_1778_ = _e1596_;
                                        phi_1779_ = _e1598_;
                                        phi_1780_ = _e1600_;
                                        phi_1781_ = _e1602_;
                                        phi_1782_ = _e1604_;
                                        phi_1783_ = _e1606_;
                                    } else {
                                        phi_1778_ = _e1215_;
                                        phi_1779_ = _e1217_;
                                        phi_1780_ = _e1219_;
                                        phi_1781_ = _e1221_;
                                        phi_1782_ = _e1223_;
                                        phi_1783_ = _e1225_;
                                    }
                                    let _e1608_ = phi_1778_;
                                    let _e1610_ = phi_1779_;
                                    let _e1612_ = phi_1780_;
                                    let _e1614_ = phi_1781_;
                                    let _e1616_ = phi_1782_;
                                    let _e1618_ = phi_1783_;
                                    phi_1149_ = _e1608_;
                                    phi_1151_ = _e1610_;
                                    phi_1153_ = _e1612_;
                                    phi_1155_ = _e1614_;
                                    phi_1157_ = _e1616_;
                                    phi_1159_ = _e1618_;
                                } else {
                                    phi_1149_ = vec4<f32>();
                                    phi_1151_ = vec4<f32>();
                                    phi_1153_ = vec4<f32>();
                                    phi_1155_ = vec4<f32>();
                                    phi_1157_ = vec4<f32>();
                                    phi_1159_ = vec4<f32>();
                                }
                                let _e1620_ = phi_1149_;
                                let _e1622_ = phi_1151_;
                                let _e1624_ = phi_1153_;
                                let _e1626_ = phi_1155_;
                                let _e1628_ = phi_1157_;
                                let _e1630_ = phi_1159_;
                                phi_1145_ = _e1238_;
                                phi_1148_ = _e1620_;
                                phi_1150_ = _e1622_;
                                phi_1152_ = _e1624_;
                                phi_1154_ = _e1626_;
                                phi_1156_ = _e1628_;
                                phi_1158_ = _e1630_;
                                phi_4361_ = false;
                                if !(_e1244_) {
                                    break;
                                }
                            }
                            let _e1633_ = phi_4361_;
                            phi_4427_ = _e1633_;
                            if _e1633_ {
                                break;
                            }
                            let _e1635_ = local_4_;
                            let _e1637_ = bitcast<u32>(_e1635_.x);
                            let _e1638_ = (_e1637_ & 2139095040u);
                            if (_e1638_ == 2139095040u) {
                                phi_1795_ = true;
                            } else {
                                phi_1795_ = (_e1638_ == 0u);
                            }
                            let _e1642_ = phi_1795_;
                            let _e1646_ = local_5_;
                            let _e1648_ = bitcast<u32>(_e1646_.y);
                            let _e1649_ = (_e1648_ & 2139095040u);
                            if (_e1649_ == 2139095040u) {
                                phi_1812_ = true;
                            } else {
                                phi_1812_ = (_e1649_ == 0u);
                            }
                            let _e1653_ = phi_1812_;
                            let _e1657_ = local_6_;
                            let _e1659_ = bitcast<u32>(_e1657_.z);
                            let _e1660_ = (_e1659_ & 2139095040u);
                            if (_e1660_ == 2139095040u) {
                                phi_1829_ = true;
                            } else {
                                phi_1829_ = (_e1660_ == 0u);
                            }
                            let _e1664_ = phi_1829_;
                            let _e1668_ = local_7_;
                            let _e1670_ = bitcast<u32>(_e1668_.w);
                            let _e1671_ = (_e1670_ & 2139095040u);
                            if (_e1671_ == 2139095040u) {
                                phi_1846_ = true;
                            } else {
                                phi_1846_ = (_e1671_ == 0u);
                            }
                            let _e1675_ = phi_1846_;
                            if _e1170_ {
                                if _e1170_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1169_] = vec4<f32>(bitcast<f32>(select(_e1637_, 0u, _e1642_)), bitcast<f32>(select(_e1648_, 0u, _e1653_)), bitcast<f32>(select(_e1659_, 0u, _e1664_)), bitcast<f32>(select(_e1670_, 0u, _e1675_)));
                            }
                            let _e1682_ = local_8_;
                            let _e1684_ = bitcast<u32>(_e1682_.x);
                            let _e1685_ = (_e1684_ & 2139095040u);
                            if (_e1685_ == 2139095040u) {
                                phi_1875_ = true;
                            } else {
                                phi_1875_ = (_e1685_ == 0u);
                            }
                            let _e1689_ = phi_1875_;
                            let _e1693_ = local_9_;
                            let _e1695_ = bitcast<u32>(_e1693_.y);
                            let _e1696_ = (_e1695_ & 2139095040u);
                            if (_e1696_ == 2139095040u) {
                                phi_1892_ = true;
                            } else {
                                phi_1892_ = (_e1696_ == 0u);
                            }
                            let _e1700_ = phi_1892_;
                            let _e1704_ = local_10_;
                            let _e1706_ = bitcast<u32>(_e1704_.z);
                            let _e1707_ = (_e1706_ & 2139095040u);
                            if (_e1707_ == 2139095040u) {
                                phi_1909_ = true;
                            } else {
                                phi_1909_ = (_e1707_ == 0u);
                            }
                            let _e1711_ = phi_1909_;
                            let _e1715_ = local_11_;
                            let _e1717_ = bitcast<u32>(_e1715_.w);
                            let _e1718_ = (_e1717_ & 2139095040u);
                            if (_e1718_ == 2139095040u) {
                                phi_1926_ = true;
                            } else {
                                phi_1926_ = (_e1718_ == 0u);
                            }
                            let _e1722_ = phi_1926_;
                            if _e1177_ {
                                if _e1177_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1176_] = vec4<f32>(bitcast<f32>(select(_e1684_, 0u, _e1689_)), bitcast<f32>(select(_e1695_, 0u, _e1700_)), bitcast<f32>(select(_e1706_, 0u, _e1711_)), bitcast<f32>(select(_e1717_, 0u, _e1722_)));
                            }
                            let _e1729_ = local_12_;
                            let _e1731_ = bitcast<u32>(_e1729_.x);
                            let _e1732_ = (_e1731_ & 2139095040u);
                            if (_e1732_ == 2139095040u) {
                                phi_1955_ = true;
                            } else {
                                phi_1955_ = (_e1732_ == 0u);
                            }
                            let _e1736_ = phi_1955_;
                            let _e1740_ = local_13_;
                            let _e1742_ = bitcast<u32>(_e1740_.y);
                            let _e1743_ = (_e1742_ & 2139095040u);
                            if (_e1743_ == 2139095040u) {
                                phi_1972_ = true;
                            } else {
                                phi_1972_ = (_e1743_ == 0u);
                            }
                            let _e1747_ = phi_1972_;
                            let _e1751_ = local_14_;
                            let _e1753_ = bitcast<u32>(_e1751_.z);
                            let _e1754_ = (_e1753_ & 2139095040u);
                            if (_e1754_ == 2139095040u) {
                                phi_1989_ = true;
                            } else {
                                phi_1989_ = (_e1754_ == 0u);
                            }
                            let _e1758_ = phi_1989_;
                            let _e1762_ = local_15_;
                            let _e1764_ = bitcast<u32>(_e1762_.w);
                            let _e1765_ = (_e1764_ & 2139095040u);
                            if (_e1765_ == 2139095040u) {
                                phi_2006_ = true;
                            } else {
                                phi_2006_ = (_e1765_ == 0u);
                            }
                            let _e1769_ = phi_2006_;
                            if _e1184_ {
                                if _e1184_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1183_] = vec4<f32>(bitcast<f32>(select(_e1731_, 0u, _e1736_)), bitcast<f32>(select(_e1742_, 0u, _e1747_)), bitcast<f32>(select(_e1753_, 0u, _e1758_)), bitcast<f32>(select(_e1764_, 0u, _e1769_)));
                            }
                            let _e1776_ = local_16_;
                            let _e1778_ = bitcast<u32>(_e1776_.x);
                            let _e1779_ = (_e1778_ & 2139095040u);
                            if (_e1779_ == 2139095040u) {
                                phi_2035_ = true;
                            } else {
                                phi_2035_ = (_e1779_ == 0u);
                            }
                            let _e1783_ = phi_2035_;
                            let _e1787_ = local_17_;
                            let _e1789_ = bitcast<u32>(_e1787_.y);
                            let _e1790_ = (_e1789_ & 2139095040u);
                            if (_e1790_ == 2139095040u) {
                                phi_2052_ = true;
                            } else {
                                phi_2052_ = (_e1790_ == 0u);
                            }
                            let _e1794_ = phi_2052_;
                            let _e1798_ = local_18_;
                            let _e1800_ = bitcast<u32>(_e1798_.z);
                            let _e1801_ = (_e1800_ & 2139095040u);
                            if (_e1801_ == 2139095040u) {
                                phi_2069_ = true;
                            } else {
                                phi_2069_ = (_e1801_ == 0u);
                            }
                            let _e1805_ = phi_2069_;
                            let _e1809_ = local_19_;
                            let _e1811_ = bitcast<u32>(_e1809_.w);
                            let _e1812_ = (_e1811_ & 2139095040u);
                            if (_e1812_ == 2139095040u) {
                                phi_2086_ = true;
                            } else {
                                phi_2086_ = (_e1812_ == 0u);
                            }
                            let _e1816_ = phi_2086_;
                            if _e1191_ {
                                if _e1191_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1190_] = vec4<f32>(bitcast<f32>(select(_e1778_, 0u, _e1783_)), bitcast<f32>(select(_e1789_, 0u, _e1794_)), bitcast<f32>(select(_e1800_, 0u, _e1805_)), bitcast<f32>(select(_e1811_, 0u, _e1816_)));
                            }
                            let _e1823_ = local_20_;
                            let _e1825_ = bitcast<u32>(_e1823_.x);
                            let _e1826_ = (_e1825_ & 2139095040u);
                            if (_e1826_ == 2139095040u) {
                                phi_2115_ = true;
                            } else {
                                phi_2115_ = (_e1826_ == 0u);
                            }
                            let _e1830_ = phi_2115_;
                            let _e1834_ = local_21_;
                            let _e1836_ = bitcast<u32>(_e1834_.y);
                            let _e1837_ = (_e1836_ & 2139095040u);
                            if (_e1837_ == 2139095040u) {
                                phi_2132_ = true;
                            } else {
                                phi_2132_ = (_e1837_ == 0u);
                            }
                            let _e1841_ = phi_2132_;
                            let _e1845_ = local_22_;
                            let _e1847_ = bitcast<u32>(_e1845_.z);
                            let _e1848_ = (_e1847_ & 2139095040u);
                            if (_e1848_ == 2139095040u) {
                                phi_2149_ = true;
                            } else {
                                phi_2149_ = (_e1848_ == 0u);
                            }
                            let _e1852_ = phi_2149_;
                            let _e1856_ = local_23_;
                            let _e1858_ = bitcast<u32>(_e1856_.w);
                            let _e1859_ = (_e1858_ & 2139095040u);
                            if (_e1859_ == 2139095040u) {
                                phi_2166_ = true;
                            } else {
                                phi_2166_ = (_e1859_ == 0u);
                            }
                            let _e1863_ = phi_2166_;
                            if _e1198_ {
                                if _e1198_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1197_] = vec4<f32>(bitcast<f32>(select(_e1825_, 0u, _e1830_)), bitcast<f32>(select(_e1836_, 0u, _e1841_)), bitcast<f32>(select(_e1847_, 0u, _e1852_)), bitcast<f32>(select(_e1858_, 0u, _e1863_)));
                            }
                            let _e1870_ = local_24_;
                            let _e1872_ = bitcast<u32>(_e1870_.x);
                            let _e1873_ = (_e1872_ & 2139095040u);
                            if (_e1873_ == 2139095040u) {
                                phi_2195_ = true;
                            } else {
                                phi_2195_ = (_e1873_ == 0u);
                            }
                            let _e1877_ = phi_2195_;
                            let _e1881_ = local_25_;
                            let _e1883_ = bitcast<u32>(_e1881_.y);
                            let _e1884_ = (_e1883_ & 2139095040u);
                            if (_e1884_ == 2139095040u) {
                                phi_2212_ = true;
                            } else {
                                phi_2212_ = (_e1884_ == 0u);
                            }
                            let _e1888_ = phi_2212_;
                            let _e1892_ = local_26_;
                            let _e1894_ = bitcast<u32>(_e1892_.z);
                            let _e1895_ = (_e1894_ & 2139095040u);
                            if (_e1895_ == 2139095040u) {
                                phi_2229_ = true;
                            } else {
                                phi_2229_ = (_e1895_ == 0u);
                            }
                            let _e1899_ = phi_2229_;
                            let _e1903_ = local_27_;
                            let _e1905_ = bitcast<u32>(_e1903_.w);
                            let _e1906_ = (_e1905_ & 2139095040u);
                            if (_e1906_ == 2139095040u) {
                                phi_2246_ = true;
                            } else {
                                phi_2246_ = (_e1906_ == 0u);
                            }
                            let _e1910_ = phi_2246_;
                            if _e1205_ {
                                if _e1205_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1204_] = vec4<f32>(bitcast<f32>(select(_e1872_, 0u, _e1877_)), bitcast<f32>(select(_e1883_, 0u, _e1888_)), bitcast<f32>(select(_e1894_, 0u, _e1899_)), bitcast<f32>(select(_e1905_, 0u, _e1910_)));
                            }
                            phi_4432_ = _e1633_;
                        }
                        let _e1917_ = phi_4432_;
                        phi_4431_ = _e1917_;
                    }
                    let _e1919_ = phi_4431_;
                    phi_4427_ = _e1919_;
                    break;
                }
                case 2: {
                    let _e303_ = uniforms.member.member;
                    let _e306_ = uniforms.member.member_1_;
                    let _e308_ = select(1u, _e306_, (1u < _e306_));
                    let _e311_ = uniforms.member.member_2_;
                    let _e314_ = uniforms.member.member_6_;
                    let _e316_ = select(1u, _e314_, (1u < _e314_));
                    if (_e55_.x >= _e311_) {
                        phi_4429_ = false;
                    } else {
                        if (_e55_.y >= _e308_) {
                            phi_4430_ = false;
                        } else {
                            let _e319_ = (_e55_.x * 13u);
                            let _e320_ = (_e319_ < _e61_);
                            if _e320_ {
                                if _e320_ {
                                } else {
                                    phi_4427_ = true;
                                    break;
                                }
                                let _e323_ = tile_descriptors.member[_e319_];
                                phi_2304_ = _e323_;
                            } else {
                                phi_2304_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e325_ = phi_2304_;
                            let _e327_ = bitcast<u32>(_e325_.x);
                            let _e329_ = bitcast<u32>(_e325_.y);
                            if (_e327_ < _e316_) {
                                let _e331_ = (_e329_ < _e316_);
                                let _e333_ = ((_e55_.y * 256u) + _e68_);
                                if _e331_ {
                                    let _e334_ = (_e333_ < _e303_);
                                    if _e334_ {
                                        let _e335_ = (_e55_.z * _e316_);
                                        let _e339_ = ((((_e335_ + _e327_) * _e303_) + _e333_) * 6u);
                                        let _e343_ = ((((_e335_ + _e329_) * _e303_) + _e333_) * 2u);
                                        let _e344_ = (_e339_ < _e63_);
                                        if _e344_ {
                                            if _e344_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e347_ = feature_outputs.member[_e339_];
                                            phi_2344_ = _e347_;
                                        } else {
                                            phi_2344_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e349_ = phi_2344_;
                                        let _e350_ = (_e339_ + 1u);
                                        let _e351_ = (_e350_ < _e63_);
                                        if _e351_ {
                                            if _e351_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e354_ = feature_outputs.member[_e350_];
                                            phi_2357_ = _e354_;
                                        } else {
                                            phi_2357_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e356_ = phi_2357_;
                                        let _e357_ = (_e339_ + 2u);
                                        let _e358_ = (_e357_ < _e63_);
                                        if _e358_ {
                                            if _e358_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e361_ = feature_outputs.member[_e357_];
                                            phi_2370_ = _e361_;
                                        } else {
                                            phi_2370_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e363_ = phi_2370_;
                                        let _e364_ = (_e339_ + 3u);
                                        let _e365_ = (_e364_ < _e63_);
                                        if _e365_ {
                                            if _e365_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e368_ = feature_outputs.member[_e364_];
                                            phi_2383_ = _e368_;
                                        } else {
                                            phi_2383_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e370_ = phi_2383_;
                                        let _e371_ = (_e339_ + 4u);
                                        let _e372_ = (_e371_ < _e63_);
                                        if _e372_ {
                                            if _e372_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e375_ = feature_outputs.member[_e371_];
                                            phi_2396_ = _e375_;
                                        } else {
                                            phi_2396_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e377_ = phi_2396_;
                                        let _e378_ = (_e339_ + 5u);
                                        let _e379_ = (_e378_ < _e63_);
                                        if _e379_ {
                                            if _e379_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e382_ = feature_outputs.member[_e378_];
                                            phi_2409_ = _e382_;
                                        } else {
                                            phi_2409_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e384_ = phi_2409_;
                                        let _e385_ = (_e343_ < _e59_);
                                        if _e385_ {
                                            if _e385_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e388_ = block_inputs_or_d_loss_d_inputs.member[_e343_];
                                            phi_2421_ = _e388_;
                                        } else {
                                            phi_2421_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e390_ = phi_2421_;
                                        let _e391_ = (_e343_ + 1u);
                                        let _e392_ = (_e391_ < _e59_);
                                        if _e392_ {
                                            if _e392_ {
                                            } else {
                                                phi_4427_ = true;
                                                break;
                                            }
                                            let _e395_ = block_inputs_or_d_loss_d_inputs.member[_e391_];
                                            phi_2434_ = _e395_;
                                        } else {
                                            phi_2434_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e397_ = phi_2434_;
                                        phi_2443_ = _e397_.w;
                                        phi_2444_ = _e397_.z;
                                        phi_2445_ = _e397_.y;
                                        phi_2446_ = _e397_.x;
                                        phi_2447_ = _e390_.w;
                                        phi_2448_ = _e390_.z;
                                        phi_2449_ = _e390_.y;
                                        phi_2450_ = _e390_.x;
                                        phi_2451_ = _e384_;
                                        phi_2452_ = _e377_;
                                        phi_2453_ = _e370_;
                                        phi_2454_ = _e363_;
                                        phi_2455_ = _e356_;
                                        phi_2456_ = _e349_;
                                    } else {
                                        phi_2443_ = f32();
                                        phi_2444_ = f32();
                                        phi_2445_ = f32();
                                        phi_2446_ = f32();
                                        phi_2447_ = f32();
                                        phi_2448_ = f32();
                                        phi_2449_ = f32();
                                        phi_2450_ = f32();
                                        phi_2451_ = vec4<f32>();
                                        phi_2452_ = vec4<f32>();
                                        phi_2453_ = vec4<f32>();
                                        phi_2454_ = vec4<f32>();
                                        phi_2455_ = vec4<f32>();
                                        phi_2456_ = vec4<f32>();
                                    }
                                    let _e407_ = phi_2443_;
                                    let _e409_ = phi_2444_;
                                    let _e411_ = phi_2445_;
                                    let _e413_ = phi_2446_;
                                    let _e415_ = phi_2447_;
                                    let _e417_ = phi_2448_;
                                    let _e419_ = phi_2449_;
                                    let _e421_ = phi_2450_;
                                    let _e423_ = phi_2451_;
                                    let _e425_ = phi_2452_;
                                    let _e427_ = phi_2453_;
                                    let _e429_ = phi_2454_;
                                    let _e431_ = phi_2455_;
                                    let _e433_ = phi_2456_;
                                    phi_2458_ = _e407_;
                                    phi_2459_ = _e409_;
                                    phi_2460_ = _e411_;
                                    phi_2461_ = _e413_;
                                    phi_2462_ = _e415_;
                                    phi_2463_ = _e417_;
                                    phi_2464_ = _e419_;
                                    phi_2465_ = _e421_;
                                    phi_2466_ = _e423_;
                                    phi_2467_ = _e425_;
                                    phi_2468_ = _e427_;
                                    phi_2469_ = _e429_;
                                    phi_2470_ = _e431_;
                                    phi_2471_ = _e433_;
                                    phi_2473_ = select(true, false, _e334_);
                                } else {
                                    phi_2458_ = f32();
                                    phi_2459_ = f32();
                                    phi_2460_ = f32();
                                    phi_2461_ = f32();
                                    phi_2462_ = f32();
                                    phi_2463_ = f32();
                                    phi_2464_ = f32();
                                    phi_2465_ = f32();
                                    phi_2466_ = vec4<f32>();
                                    phi_2467_ = vec4<f32>();
                                    phi_2468_ = vec4<f32>();
                                    phi_2469_ = vec4<f32>();
                                    phi_2470_ = vec4<f32>();
                                    phi_2471_ = vec4<f32>();
                                    phi_2473_ = false;
                                }
                                let _e436_ = phi_2458_;
                                let _e438_ = phi_2459_;
                                let _e440_ = phi_2460_;
                                let _e442_ = phi_2461_;
                                let _e444_ = phi_2462_;
                                let _e446_ = phi_2463_;
                                let _e448_ = phi_2464_;
                                let _e450_ = phi_2465_;
                                let _e452_ = phi_2466_;
                                let _e454_ = phi_2467_;
                                let _e456_ = phi_2468_;
                                let _e458_ = phi_2469_;
                                let _e460_ = phi_2470_;
                                let _e462_ = phi_2471_;
                                let _e464_ = phi_2473_;
                                phi_2474_ = _e436_;
                                phi_2475_ = _e438_;
                                phi_2476_ = _e440_;
                                phi_2477_ = _e442_;
                                phi_2478_ = _e444_;
                                phi_2479_ = _e446_;
                                phi_2480_ = _e448_;
                                phi_2481_ = _e450_;
                                phi_2482_ = _e452_;
                                phi_2483_ = _e454_;
                                phi_2484_ = _e456_;
                                phi_2485_ = _e458_;
                                phi_2486_ = _e460_;
                                phi_2487_ = _e462_;
                                phi_2488_ = select(true, false, _e331_);
                                phi_2489_ = _e464_;
                            } else {
                                phi_2474_ = f32();
                                phi_2475_ = f32();
                                phi_2476_ = f32();
                                phi_2477_ = f32();
                                phi_2478_ = f32();
                                phi_2479_ = f32();
                                phi_2480_ = f32();
                                phi_2481_ = f32();
                                phi_2482_ = vec4<f32>();
                                phi_2483_ = vec4<f32>();
                                phi_2484_ = vec4<f32>();
                                phi_2485_ = vec4<f32>();
                                phi_2486_ = vec4<f32>();
                                phi_2487_ = vec4<f32>();
                                phi_2488_ = true;
                                phi_2489_ = false;
                            }
                            let _e467_ = phi_2474_;
                            let _e469_ = phi_2475_;
                            let _e471_ = phi_2476_;
                            let _e473_ = phi_2477_;
                            let _e475_ = phi_2478_;
                            let _e477_ = phi_2479_;
                            let _e479_ = phi_2480_;
                            let _e481_ = phi_2481_;
                            let _e483_ = phi_2482_;
                            let _e485_ = phi_2483_;
                            let _e487_ = phi_2484_;
                            let _e489_ = phi_2485_;
                            let _e491_ = phi_2486_;
                            let _e493_ = phi_2487_;
                            let _e495_ = phi_2488_;
                            let _e497_ = phi_2489_;
                            let _e498_ = select(_e497_, true, _e495_);
                            let _e499_ = select(_e467_, 0f, _e498_);
                            let _e500_ = select(_e469_, 0f, _e498_);
                            let _e501_ = select(_e471_, 0f, _e498_);
                            let _e502_ = select(_e473_, 0f, _e498_);
                            let _e503_ = select(_e475_, 0f, _e498_);
                            let _e504_ = select(_e477_, 0f, _e498_);
                            let _e505_ = select(_e479_, 0f, _e498_);
                            let _e506_ = select(_e481_, 0f, _e498_);
                            let _e507_ = vec4(_e498_);
                            let _e508_ = select(_e483_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e509_ = select(_e485_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e510_ = select(_e487_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e511_ = select(_e489_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e512_ = select(_e491_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e513_ = select(_e493_, vec4<f32>(0f, 0f, 0f, 0f), _e507_);
                            let _e519_ = ((_e311_ + ((((_e55_.z * _e311_) + _e55_.x) * _e308_) + _e55_.y)) * 48u);
                            phi_4233_ = false;
                            phi_2527_ = type_19_(0u, 4u);
                            loop {
                                let _e521_ = phi_4233_;
                                let _e523_ = phi_2527_;
                                if (_e523_.member < _e523_.member_1_) {
                                    phi_2528_ = type_19_((_e523_.member + 1u), _e523_.member_1_);
                                    phi_2551_ = type_19_(1u, _e523_.member);
                                } else {
                                    phi_2528_ = _e523_;
                                    phi_2551_ = type_19_(0u, u32());
                                }
                                let _e536_ = phi_2528_;
                                let _e538_ = phi_2551_;
                                let _e542_ = (bitcast<i32>(_e538_.member) != 0i);
                                if _e542_ {
                                    let _e543_ = (_e538_.member_1_ & 3u);
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_2569_ = _e511_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2569_ = _e511_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2569_ = _e511_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2569_ = _e511_.w;
                                            break;
                                        }
                                    }
                                    let _e550_ = phi_2569_;
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_2580_ = _e510_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2580_ = _e510_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2580_ = _e510_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2580_ = _e510_.w;
                                            break;
                                        }
                                    }
                                    let _e557_ = phi_2580_;
                                    let _e564_ = bitcast<u32>(((_e550_ * _e506_) + (_e557_ * _e502_)));
                                    let _e565_ = (_e564_ & 2139095040u);
                                    if (_e565_ == 2139095040u) {
                                        phi_2597_ = true;
                                    } else {
                                        phi_2597_ = (_e565_ == 0u);
                                    }
                                    let _e569_ = phi_2597_;
                                    let _e572_ = bitcast<u32>(((_e550_ * _e505_) + (_e557_ * _e501_)));
                                    let _e573_ = (_e572_ & 2139095040u);
                                    if (_e573_ == 2139095040u) {
                                        phi_2613_ = true;
                                    } else {
                                        phi_2613_ = (_e573_ == 0u);
                                    }
                                    let _e577_ = phi_2613_;
                                    let _e2887 = subgroupAdd(bitcast<f32>(select(_e564_, 0u, _e569_)));
                                    let _e2891 = subgroupAdd(bitcast<f32>(select(_e572_, 0u, _e577_)));
                                    let _e582_ = (_e57_ == 0u);
                                    let _e586_ = select(255u, _e68_, (_e68_ < 255u));
                                    let _e587_ = (_e586_ < 256u);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e2887, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e2891, _e582_);
                                    workgroupBarrier();
                                    phi_2651_ = 128u;
                                    loop {
                                        let _e591_ = phi_2651_;
                                        let _e592_ = (_e591_ > 0u);
                                        if _e592_ {
                                            if (_e68_ < _e591_) {
                                                let _e594_ = (_e68_ + _e591_);
                                                let _e595_ = (_e594_ < 256u);
                                                if _e595_ {
                                                } else {
                                                    phi_4220_ = true;
                                                    break;
                                                }
                                                let _e597_ = reduce_a[_e594_];
                                                let _e598_ = (_e68_ < 256u);
                                                if _e598_ {
                                                } else {
                                                    phi_4220_ = true;
                                                    break;
                                                }
                                                let _e600_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e600_ + _e597_);
                                                if _e595_ {
                                                } else {
                                                    phi_4220_ = true;
                                                    break;
                                                }
                                                let _e603_ = reduce_b[_e594_];
                                                if _e598_ {
                                                } else {
                                                    phi_4220_ = true;
                                                    break;
                                                }
                                                let _e605_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e605_ + _e603_);
                                            }
                                            workgroupBarrier();
                                            phi_2652_ = (_e591_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2652_ = u32();
                                        }
                                        let _e610_ = phi_2652_;
                                        phi_2651_ = _e610_;
                                        phi_4220_ = _e521_;
                                        if !(_e592_) {
                                            break;
                                        }
                                    }
                                    let _e613_ = phi_4220_;
                                    phi_4330_ = _e613_;
                                    if _e613_ {
                                        break;
                                    }
                                    let _e615_ = reduce_a[0];
                                    let _e616_ = bitcast<u32>(_e615_);
                                    let _e617_ = (_e616_ & 2139095040u);
                                    if (_e617_ == 2139095040u) {
                                        phi_2712_ = true;
                                    } else {
                                        phi_2712_ = (_e617_ == 0u);
                                    }
                                    let _e621_ = phi_2712_;
                                    let _e625_ = reduce_b[0];
                                    let _e626_ = bitcast<u32>(_e625_);
                                    let _e627_ = (_e626_ & 2139095040u);
                                    if (_e627_ == 2139095040u) {
                                        phi_2730_ = true;
                                    } else {
                                        phi_2730_ = (_e627_ == 0u);
                                    }
                                    let _e631_ = phi_2730_;
                                    workgroupBarrier();
                                    let _e640_ = bitcast<u32>(((_e550_ * _e504_) + (_e557_ * _e500_)));
                                    let _e641_ = (_e640_ & 2139095040u);
                                    if (_e641_ == 2139095040u) {
                                        phi_2752_ = true;
                                    } else {
                                        phi_2752_ = (_e641_ == 0u);
                                    }
                                    let _e645_ = phi_2752_;
                                    let _e648_ = bitcast<u32>(((_e550_ * _e503_) + (_e557_ * _e499_)));
                                    let _e649_ = (_e648_ & 2139095040u);
                                    if (_e649_ == 2139095040u) {
                                        phi_2768_ = true;
                                    } else {
                                        phi_2768_ = (_e649_ == 0u);
                                    }
                                    let _e653_ = phi_2768_;
                                    let _e3000 = subgroupAdd(bitcast<f32>(select(_e640_, 0u, _e645_)));
                                    let _e3004 = subgroupAdd(bitcast<f32>(select(_e648_, 0u, _e653_)));
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e3000, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e3004, _e582_);
                                    workgroupBarrier();
                                    phi_2806_ = 128u;
                                    loop {
                                        let _e661_ = phi_2806_;
                                        let _e662_ = (_e661_ > 0u);
                                        if _e662_ {
                                            if (_e68_ < _e661_) {
                                                let _e664_ = (_e68_ + _e661_);
                                                let _e665_ = (_e664_ < 256u);
                                                if _e665_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e667_ = reduce_a[_e664_];
                                                let _e668_ = (_e68_ < 256u);
                                                if _e668_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e670_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e670_ + _e667_);
                                                if _e665_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e673_ = reduce_b[_e664_];
                                                if _e668_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e675_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e675_ + _e673_);
                                            }
                                            workgroupBarrier();
                                            phi_2807_ = (_e661_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2807_ = u32();
                                        }
                                        let _e680_ = phi_2807_;
                                        phi_2806_ = _e680_;
                                        phi_4251_ = _e613_;
                                        if !(_e662_) {
                                            break;
                                        }
                                    }
                                    let _e683_ = phi_4251_;
                                    phi_4330_ = _e683_;
                                    if _e683_ {
                                        break;
                                    }
                                    let _e684_ = reduce_a[0];
                                    let _e685_ = bitcast<u32>(_e684_);
                                    let _e686_ = (_e685_ & 2139095040u);
                                    if (_e686_ == 2139095040u) {
                                        phi_2867_ = true;
                                    } else {
                                        phi_2867_ = (_e686_ == 0u);
                                    }
                                    let _e690_ = phi_2867_;
                                    let _e693_ = reduce_b[0];
                                    let _e694_ = bitcast<u32>(_e693_);
                                    let _e695_ = (_e694_ & 2139095040u);
                                    if (_e695_ == 2139095040u) {
                                        phi_2885_ = true;
                                    } else {
                                        phi_2885_ = (_e695_ == 0u);
                                    }
                                    let _e699_ = phi_2885_;
                                    workgroupBarrier();
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_2901_ = _e509_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2901_ = _e509_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2901_ = _e509_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2901_ = _e509_.w;
                                            break;
                                        }
                                    }
                                    let _e708_ = phi_2901_;
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_2912_ = _e508_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2912_ = _e508_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2912_ = _e508_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2912_ = _e508_.w;
                                            break;
                                        }
                                    }
                                    let _e715_ = phi_2912_;
                                    let _e722_ = bitcast<u32>(((_e708_ * _e506_) + (_e715_ * _e502_)));
                                    let _e723_ = (_e722_ & 2139095040u);
                                    if (_e723_ == 2139095040u) {
                                        phi_2929_ = true;
                                    } else {
                                        phi_2929_ = (_e723_ == 0u);
                                    }
                                    let _e727_ = phi_2929_;
                                    let _e730_ = bitcast<u32>(((_e708_ * _e505_) + (_e715_ * _e501_)));
                                    let _e731_ = (_e730_ & 2139095040u);
                                    if (_e731_ == 2139095040u) {
                                        phi_2945_ = true;
                                    } else {
                                        phi_2945_ = (_e731_ == 0u);
                                    }
                                    let _e735_ = phi_2945_;
                                    let _e3117 = subgroupAdd(bitcast<f32>(select(_e722_, 0u, _e727_)));
                                    let _e3121 = subgroupAdd(bitcast<f32>(select(_e730_, 0u, _e735_)));
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e3117, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e3121, _e582_);
                                    workgroupBarrier();
                                    phi_2983_ = 128u;
                                    loop {
                                        let _e743_ = phi_2983_;
                                        let _e744_ = (_e743_ > 0u);
                                        if _e744_ {
                                            if (_e68_ < _e743_) {
                                                let _e746_ = (_e68_ + _e743_);
                                                let _e747_ = (_e746_ < 256u);
                                                if _e747_ {
                                                } else {
                                                    phi_4266_ = true;
                                                    break;
                                                }
                                                let _e749_ = reduce_a[_e746_];
                                                let _e750_ = (_e68_ < 256u);
                                                if _e750_ {
                                                } else {
                                                    phi_4266_ = true;
                                                    break;
                                                }
                                                let _e752_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e752_ + _e749_);
                                                if _e747_ {
                                                } else {
                                                    phi_4266_ = true;
                                                    break;
                                                }
                                                let _e755_ = reduce_b[_e746_];
                                                if _e750_ {
                                                } else {
                                                    phi_4266_ = true;
                                                    break;
                                                }
                                                let _e757_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e757_ + _e755_);
                                            }
                                            workgroupBarrier();
                                            phi_2984_ = (_e743_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2984_ = u32();
                                        }
                                        let _e762_ = phi_2984_;
                                        phi_2983_ = _e762_;
                                        phi_4266_ = _e683_;
                                        if !(_e744_) {
                                            break;
                                        }
                                    }
                                    let _e765_ = phi_4266_;
                                    phi_4330_ = _e765_;
                                    if _e765_ {
                                        break;
                                    }
                                    let _e766_ = reduce_a[0];
                                    let _e767_ = bitcast<u32>(_e766_);
                                    let _e768_ = (_e767_ & 2139095040u);
                                    if (_e768_ == 2139095040u) {
                                        phi_3044_ = true;
                                    } else {
                                        phi_3044_ = (_e768_ == 0u);
                                    }
                                    let _e772_ = phi_3044_;
                                    let _e775_ = reduce_b[0];
                                    let _e776_ = bitcast<u32>(_e775_);
                                    let _e777_ = (_e776_ & 2139095040u);
                                    if (_e777_ == 2139095040u) {
                                        phi_3062_ = true;
                                    } else {
                                        phi_3062_ = (_e777_ == 0u);
                                    }
                                    let _e781_ = phi_3062_;
                                    workgroupBarrier();
                                    let _e790_ = bitcast<u32>(((_e708_ * _e504_) + (_e715_ * _e500_)));
                                    let _e791_ = (_e790_ & 2139095040u);
                                    if (_e791_ == 2139095040u) {
                                        phi_3084_ = true;
                                    } else {
                                        phi_3084_ = (_e791_ == 0u);
                                    }
                                    let _e795_ = phi_3084_;
                                    let _e798_ = bitcast<u32>(((_e708_ * _e503_) + (_e715_ * _e499_)));
                                    let _e799_ = (_e798_ & 2139095040u);
                                    if (_e799_ == 2139095040u) {
                                        phi_3100_ = true;
                                    } else {
                                        phi_3100_ = (_e799_ == 0u);
                                    }
                                    let _e803_ = phi_3100_;
                                    let _e3222 = subgroupAdd(bitcast<f32>(select(_e790_, 0u, _e795_)));
                                    let _e3226 = subgroupAdd(bitcast<f32>(select(_e798_, 0u, _e803_)));
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e3222, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e3226, _e582_);
                                    workgroupBarrier();
                                    phi_3138_ = 128u;
                                    loop {
                                        let _e811_ = phi_3138_;
                                        let _e812_ = (_e811_ > 0u);
                                        if _e812_ {
                                            if (_e68_ < _e811_) {
                                                let _e814_ = (_e68_ + _e811_);
                                                let _e815_ = (_e814_ < 256u);
                                                if _e815_ {
                                                } else {
                                                    phi_4283_ = true;
                                                    break;
                                                }
                                                let _e817_ = reduce_a[_e814_];
                                                let _e818_ = (_e68_ < 256u);
                                                if _e818_ {
                                                } else {
                                                    phi_4283_ = true;
                                                    break;
                                                }
                                                let _e820_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e820_ + _e817_);
                                                if _e815_ {
                                                } else {
                                                    phi_4283_ = true;
                                                    break;
                                                }
                                                let _e823_ = reduce_b[_e814_];
                                                if _e818_ {
                                                } else {
                                                    phi_4283_ = true;
                                                    break;
                                                }
                                                let _e825_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e825_ + _e823_);
                                            }
                                            workgroupBarrier();
                                            phi_3139_ = (_e811_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3139_ = u32();
                                        }
                                        let _e830_ = phi_3139_;
                                        phi_3138_ = _e830_;
                                        phi_4283_ = _e765_;
                                        if !(_e812_) {
                                            break;
                                        }
                                    }
                                    let _e833_ = phi_4283_;
                                    phi_4330_ = _e833_;
                                    if _e833_ {
                                        break;
                                    }
                                    let _e834_ = reduce_a[0];
                                    let _e835_ = bitcast<u32>(_e834_);
                                    let _e836_ = (_e835_ & 2139095040u);
                                    if (_e836_ == 2139095040u) {
                                        phi_3199_ = true;
                                    } else {
                                        phi_3199_ = (_e836_ == 0u);
                                    }
                                    let _e840_ = phi_3199_;
                                    let _e843_ = reduce_b[0];
                                    let _e844_ = bitcast<u32>(_e843_);
                                    let _e845_ = (_e844_ & 2139095040u);
                                    if (_e845_ == 2139095040u) {
                                        phi_3217_ = true;
                                    } else {
                                        phi_3217_ = (_e845_ == 0u);
                                    }
                                    let _e849_ = phi_3217_;
                                    workgroupBarrier();
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_3233_ = _e513_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_3233_ = _e513_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_3233_ = _e513_.z;
                                            break;
                                        }
                                        default: {
                                            phi_3233_ = _e513_.w;
                                            break;
                                        }
                                    }
                                    let _e858_ = phi_3233_;
                                    switch bitcast<i32>(_e543_) {
                                        case 0: {
                                            phi_3244_ = _e512_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_3244_ = _e512_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_3244_ = _e512_.z;
                                            break;
                                        }
                                        default: {
                                            phi_3244_ = _e512_.w;
                                            break;
                                        }
                                    }
                                    let _e865_ = phi_3244_;
                                    let _e872_ = bitcast<u32>(((_e858_ * _e506_) + (_e865_ * _e502_)));
                                    let _e873_ = (_e872_ & 2139095040u);
                                    if (_e873_ == 2139095040u) {
                                        phi_3261_ = true;
                                    } else {
                                        phi_3261_ = (_e873_ == 0u);
                                    }
                                    let _e877_ = phi_3261_;
                                    let _e880_ = bitcast<u32>(((_e858_ * _e505_) + (_e865_ * _e501_)));
                                    let _e881_ = (_e880_ & 2139095040u);
                                    if (_e881_ == 2139095040u) {
                                        phi_3277_ = true;
                                    } else {
                                        phi_3277_ = (_e881_ == 0u);
                                    }
                                    let _e885_ = phi_3277_;
                                    let _e3339 = subgroupAdd(bitcast<f32>(select(_e872_, 0u, _e877_)));
                                    let _e3343 = subgroupAdd(bitcast<f32>(select(_e880_, 0u, _e885_)));
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e3339, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e3343, _e582_);
                                    workgroupBarrier();
                                    phi_3315_ = 128u;
                                    loop {
                                        let _e893_ = phi_3315_;
                                        let _e894_ = (_e893_ > 0u);
                                        if _e894_ {
                                            if (_e68_ < _e893_) {
                                                let _e896_ = (_e68_ + _e893_);
                                                let _e897_ = (_e896_ < 256u);
                                                if _e897_ {
                                                } else {
                                                    phi_4298_ = true;
                                                    break;
                                                }
                                                let _e899_ = reduce_a[_e896_];
                                                let _e900_ = (_e68_ < 256u);
                                                if _e900_ {
                                                } else {
                                                    phi_4298_ = true;
                                                    break;
                                                }
                                                let _e902_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e902_ + _e899_);
                                                if _e897_ {
                                                } else {
                                                    phi_4298_ = true;
                                                    break;
                                                }
                                                let _e905_ = reduce_b[_e896_];
                                                if _e900_ {
                                                } else {
                                                    phi_4298_ = true;
                                                    break;
                                                }
                                                let _e907_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e907_ + _e905_);
                                            }
                                            workgroupBarrier();
                                            phi_3316_ = (_e893_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3316_ = u32();
                                        }
                                        let _e912_ = phi_3316_;
                                        phi_3315_ = _e912_;
                                        phi_4298_ = _e833_;
                                        if !(_e894_) {
                                            break;
                                        }
                                    }
                                    let _e915_ = phi_4298_;
                                    phi_4330_ = _e915_;
                                    if _e915_ {
                                        break;
                                    }
                                    let _e916_ = reduce_a[0];
                                    let _e917_ = bitcast<u32>(_e916_);
                                    let _e918_ = (_e917_ & 2139095040u);
                                    if (_e918_ == 2139095040u) {
                                        phi_3376_ = true;
                                    } else {
                                        phi_3376_ = (_e918_ == 0u);
                                    }
                                    let _e922_ = phi_3376_;
                                    let _e925_ = reduce_b[0];
                                    let _e926_ = bitcast<u32>(_e925_);
                                    let _e927_ = (_e926_ & 2139095040u);
                                    if (_e927_ == 2139095040u) {
                                        phi_3394_ = true;
                                    } else {
                                        phi_3394_ = (_e927_ == 0u);
                                    }
                                    let _e931_ = phi_3394_;
                                    workgroupBarrier();
                                    let _e940_ = bitcast<u32>(((_e858_ * _e504_) + (_e865_ * _e500_)));
                                    let _e941_ = (_e940_ & 2139095040u);
                                    if (_e941_ == 2139095040u) {
                                        phi_3416_ = true;
                                    } else {
                                        phi_3416_ = (_e941_ == 0u);
                                    }
                                    let _e945_ = phi_3416_;
                                    let _e948_ = bitcast<u32>(((_e858_ * _e503_) + (_e865_ * _e499_)));
                                    let _e949_ = (_e948_ & 2139095040u);
                                    if (_e949_ == 2139095040u) {
                                        phi_3432_ = true;
                                    } else {
                                        phi_3432_ = (_e949_ == 0u);
                                    }
                                    let _e953_ = phi_3432_;
                                    let _e3444 = subgroupAdd(bitcast<f32>(select(_e940_, 0u, _e945_)));
                                    let _e3448 = subgroupAdd(bitcast<f32>(select(_e948_, 0u, _e953_)));
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_a[_e586_] = select(0f, _e3444, _e582_);
                                    if _e587_ {
                                    } else {
                                        phi_4330_ = true;
                                        break;
                                    }
                                    reduce_b[_e586_] = select(0f, _e3448, _e582_);
                                    workgroupBarrier();
                                    phi_3470_ = 128u;
                                    loop {
                                        let _e961_ = phi_3470_;
                                        let _e962_ = (_e961_ > 0u);
                                        if _e962_ {
                                            if (_e68_ < _e961_) {
                                                let _e964_ = (_e68_ + _e961_);
                                                let _e965_ = (_e964_ < 256u);
                                                if _e965_ {
                                                } else {
                                                    phi_4315_ = true;
                                                    break;
                                                }
                                                let _e967_ = reduce_a[_e964_];
                                                let _e968_ = (_e68_ < 256u);
                                                if _e968_ {
                                                } else {
                                                    phi_4315_ = true;
                                                    break;
                                                }
                                                let _e970_ = reduce_a[_e68_];
                                                reduce_a[_e68_] = (_e970_ + _e967_);
                                                if _e965_ {
                                                } else {
                                                    phi_4315_ = true;
                                                    break;
                                                }
                                                let _e973_ = reduce_b[_e964_];
                                                if _e968_ {
                                                } else {
                                                    phi_4315_ = true;
                                                    break;
                                                }
                                                let _e975_ = reduce_b[_e68_];
                                                reduce_b[_e68_] = (_e975_ + _e973_);
                                            }
                                            workgroupBarrier();
                                            phi_3471_ = (_e961_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3471_ = u32();
                                        }
                                        let _e980_ = phi_3471_;
                                        phi_3470_ = _e980_;
                                        phi_4315_ = _e915_;
                                        if !(_e962_) {
                                            break;
                                        }
                                    }
                                    let _e983_ = phi_4315_;
                                    phi_4330_ = _e983_;
                                    if _e983_ {
                                        break;
                                    }
                                    let _e984_ = reduce_a[0];
                                    let _e985_ = bitcast<u32>(_e984_);
                                    let _e986_ = (_e985_ & 2139095040u);
                                    if (_e986_ == 2139095040u) {
                                        phi_3531_ = true;
                                    } else {
                                        phi_3531_ = (_e986_ == 0u);
                                    }
                                    let _e990_ = phi_3531_;
                                    let _e993_ = reduce_b[0];
                                    let _e994_ = bitcast<u32>(_e993_);
                                    let _e995_ = (_e994_ & 2139095040u);
                                    if (_e995_ == 2139095040u) {
                                        phi_3549_ = true;
                                    } else {
                                        phi_3549_ = (_e995_ == 0u);
                                    }
                                    let _e999_ = phi_3549_;
                                    workgroupBarrier();
                                    if (_e68_ == 0u) {
                                        let _e1003_ = (_e538_.member_1_ * 4u);
                                        let _e1004_ = (_e519_ + _e1003_);
                                        let _e1005_ = (_e65_ == 0u);
                                        if _e1005_ {
                                            phi_3571_ = 0u;
                                        } else {
                                            let _e1006_ = (_e65_ - 1u);
                                            phi_3571_ = select(_e1006_, _e1004_, (_e1004_ < _e1006_));
                                        }
                                        let _e1010_ = phi_3571_;
                                        if (_e1010_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1010_] = bitcast<f32>(select(_e616_, 0u, _e621_));
                                        let _e1014_ = (_e1004_ + 1u);
                                        if _e1005_ {
                                            phi_3590_ = 0u;
                                        } else {
                                            let _e1015_ = (_e65_ - 1u);
                                            phi_3590_ = select(_e1015_, _e1014_, (_e1014_ < _e1015_));
                                        }
                                        let _e1019_ = phi_3590_;
                                        if (_e1019_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1019_] = bitcast<f32>(select(_e626_, 0u, _e631_));
                                        let _e1023_ = (_e1004_ + 2u);
                                        if _e1005_ {
                                            phi_3609_ = 0u;
                                        } else {
                                            let _e1024_ = (_e65_ - 1u);
                                            phi_3609_ = select(_e1024_, _e1023_, (_e1023_ < _e1024_));
                                        }
                                        let _e1028_ = phi_3609_;
                                        if (_e1028_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1028_] = bitcast<f32>(select(_e685_, 0u, _e690_));
                                        let _e1032_ = (_e1004_ + 3u);
                                        if _e1005_ {
                                            phi_3628_ = 0u;
                                        } else {
                                            let _e1033_ = (_e65_ - 1u);
                                            phi_3628_ = select(_e1033_, _e1032_, (_e1032_ < _e1033_));
                                        }
                                        let _e1037_ = phi_3628_;
                                        if (_e1037_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1037_] = bitcast<f32>(select(_e694_, 0u, _e699_));
                                        let _e1042_ = ((_e519_ + 16u) + _e1003_);
                                        if _e1005_ {
                                            phi_3647_ = 0u;
                                        } else {
                                            let _e1043_ = (_e65_ - 1u);
                                            phi_3647_ = select(_e1043_, _e1042_, (_e1042_ < _e1043_));
                                        }
                                        let _e1047_ = phi_3647_;
                                        if (_e1047_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1047_] = bitcast<f32>(select(_e767_, 0u, _e772_));
                                        let _e1051_ = (_e1042_ + 1u);
                                        if _e1005_ {
                                            phi_3667_ = 0u;
                                        } else {
                                            let _e1052_ = (_e65_ - 1u);
                                            phi_3667_ = select(_e1052_, _e1051_, (_e1051_ < _e1052_));
                                        }
                                        let _e1056_ = phi_3667_;
                                        if (_e1056_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1056_] = bitcast<f32>(select(_e776_, 0u, _e781_));
                                        let _e1060_ = (_e1042_ + 2u);
                                        if _e1005_ {
                                            phi_3687_ = 0u;
                                        } else {
                                            let _e1061_ = (_e65_ - 1u);
                                            phi_3687_ = select(_e1061_, _e1060_, (_e1060_ < _e1061_));
                                        }
                                        let _e1065_ = phi_3687_;
                                        if (_e1065_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1065_] = bitcast<f32>(select(_e835_, 0u, _e840_));
                                        let _e1069_ = (_e1042_ + 3u);
                                        if _e1005_ {
                                            phi_3707_ = 0u;
                                        } else {
                                            let _e1070_ = (_e65_ - 1u);
                                            phi_3707_ = select(_e1070_, _e1069_, (_e1069_ < _e1070_));
                                        }
                                        let _e1074_ = phi_3707_;
                                        if (_e1074_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1074_] = bitcast<f32>(select(_e844_, 0u, _e849_));
                                        let _e1079_ = ((_e519_ + 32u) + _e1003_);
                                        if _e1005_ {
                                            phi_3726_ = 0u;
                                        } else {
                                            let _e1080_ = (_e65_ - 1u);
                                            phi_3726_ = select(_e1080_, _e1079_, (_e1079_ < _e1080_));
                                        }
                                        let _e1084_ = phi_3726_;
                                        if (_e1084_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1084_] = bitcast<f32>(select(_e917_, 0u, _e922_));
                                        let _e1088_ = (_e1079_ + 1u);
                                        if _e1005_ {
                                            phi_3746_ = 0u;
                                        } else {
                                            let _e1089_ = (_e65_ - 1u);
                                            phi_3746_ = select(_e1089_, _e1088_, (_e1088_ < _e1089_));
                                        }
                                        let _e1093_ = phi_3746_;
                                        if (_e1093_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1093_] = bitcast<f32>(select(_e926_, 0u, _e931_));
                                        let _e1097_ = (_e1079_ + 2u);
                                        if _e1005_ {
                                            phi_3766_ = 0u;
                                        } else {
                                            let _e1098_ = (_e65_ - 1u);
                                            phi_3766_ = select(_e1098_, _e1097_, (_e1097_ < _e1098_));
                                        }
                                        let _e1102_ = phi_3766_;
                                        if (_e1102_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1102_] = bitcast<f32>(select(_e985_, 0u, _e990_));
                                        let _e1106_ = (_e1079_ + 3u);
                                        if _e1005_ {
                                            phi_3786_ = 0u;
                                        } else {
                                            let _e1107_ = (_e65_ - 1u);
                                            phi_3786_ = select(_e1107_, _e1106_, (_e1106_ < _e1107_));
                                        }
                                        let _e1111_ = phi_3786_;
                                        if (_e1111_ < _e65_) {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1111_] = bitcast<f32>(select(_e994_, 0u, _e999_));
                                    }
                                    phi_4331_ = _e983_;
                                } else {
                                    phi_4331_ = _e521_;
                                }
                                let _e1116_ = phi_4331_;
                                phi_4233_ = _e1116_;
                                phi_2527_ = _e536_;
                                phi_4330_ = _e1116_;
                                if !(_e542_) {
                                    break;
                                }
                            }
                            let _e1119_ = phi_4330_;
                            phi_4427_ = _e1119_;
                            if _e1119_ {
                                break;
                            }
                            phi_4430_ = _e1119_;
                        }
                        let _e1121_ = phi_4430_;
                        phi_4429_ = _e1121_;
                    }
                    let _e1123_ = phi_4429_;
                    phi_4427_ = _e1123_;
                    break;
                }
                case 3: {
                    let _e76_ = uniforms.member.member_2_;
                    let _e79_ = uniforms.member.member_1_;
                    let _e81_ = select(1u, _e79_, (1u < _e79_));
                    let _e84_ = uniforms.member.member_4_;
                    if (_e55_.x >= _e76_) {
                        phi_4428_ = false;
                    } else {
                        let _e88_ = (_e55_.x * 48u);
                        phi_4185_ = false;
                        phi_3817_ = type_19_(0u, 24u);
                        loop {
                            let _e90_ = phi_4185_;
                            let _e92_ = phi_3817_;
                            if (_e92_.member < _e92_.member_1_) {
                                phi_3818_ = type_19_((_e92_.member + 1u), _e92_.member_1_);
                                phi_3841_ = type_19_(1u, _e92_.member);
                            } else {
                                phi_3818_ = _e92_;
                                phi_3841_ = type_19_(0u, u32());
                            }
                            let _e105_ = phi_3818_;
                            let _e107_ = phi_3841_;
                            let _e111_ = (bitcast<i32>(_e107_.member) != 0i);
                            if _e111_ {
                                let _e112_ = (_e107_.member_1_ * 2u);
                                let _e113_ = (_e112_ + 1u);
                                phi_4183_ = _e90_;
                                phi_3852_ = 0u;
                                phi_3855_ = 0f;
                                phi_3857_ = 0f;
                                loop {
                                    let _e115_ = phi_4183_;
                                    let _e117_ = phi_3852_;
                                    let _e119_ = phi_3855_;
                                    let _e121_ = phi_3857_;
                                    local_2_ = _e121_;
                                    local_3_ = _e119_;
                                    let _e122_ = (_e117_ < select(1u, _e84_, (1u < _e84_)));
                                    if _e122_ {
                                        phi_3866_ = _e68_;
                                        phi_3869_ = _e119_;
                                        phi_3871_ = _e121_;
                                        loop {
                                            let _e124_ = phi_3866_;
                                            let _e126_ = phi_3869_;
                                            let _e128_ = phi_3871_;
                                            local_36_ = _e126_;
                                            local_37_ = _e128_;
                                            let _e129_ = (_e124_ < _e81_);
                                            if _e129_ {
                                                let _e135_ = ((_e76_ + ((((_e117_ * _e76_) + _e55_.x) * _e81_) + _e124_)) * 48u);
                                                let _e136_ = (_e135_ + _e112_);
                                                let _e137_ = (_e65_ == 0u);
                                                if _e137_ {
                                                    phi_3896_ = 0u;
                                                } else {
                                                    let _e138_ = (_e65_ - 1u);
                                                    phi_3896_ = select(_e138_, _e136_, (_e136_ < _e138_));
                                                }
                                                let _e142_ = phi_3896_;
                                                if (_e142_ < _e65_) {
                                                } else {
                                                    phi_4180_ = true;
                                                    break;
                                                }
                                                let _e146_ = tile_weight_grads.member[_e142_];
                                                let _e148_ = (_e135_ + _e113_);
                                                if _e137_ {
                                                    phi_3916_ = 0u;
                                                } else {
                                                    let _e149_ = (_e65_ - 1u);
                                                    phi_3916_ = select(_e149_, _e148_, (_e148_ < _e149_));
                                                }
                                                let _e153_ = phi_3916_;
                                                if (_e153_ < _e65_) {
                                                } else {
                                                    phi_4180_ = true;
                                                    break;
                                                }
                                                let _e157_ = tile_weight_grads.member[_e153_];
                                                phi_3867_ = (_e124_ + 256u);
                                                phi_3870_ = (_e126_ + _e157_);
                                                phi_3872_ = (_e128_ + _e146_);
                                            } else {
                                                phi_3867_ = u32();
                                                phi_3870_ = f32();
                                                phi_3872_ = f32();
                                            }
                                            let _e161_ = phi_3867_;
                                            let _e163_ = phi_3870_;
                                            let _e165_ = phi_3872_;
                                            phi_3866_ = _e161_;
                                            phi_3869_ = _e163_;
                                            phi_3871_ = _e165_;
                                            phi_4180_ = _e115_;
                                            if !(_e129_) {
                                                break;
                                            }
                                        }
                                        let _e168_ = phi_4180_;
                                        phi_4192_ = _e168_;
                                        if _e168_ {
                                            break;
                                        }
                                        phi_4193_ = _e168_;
                                        phi_3853_ = (_e117_ + 1u);
                                        let _e2552_ = local_36_;
                                        phi_3856_ = _e2552_;
                                        let _e2555_ = local_37_;
                                        phi_3858_ = _e2555_;
                                    } else {
                                        phi_4193_ = _e115_;
                                        phi_3853_ = u32();
                                        phi_3856_ = f32();
                                        phi_3858_ = f32();
                                    }
                                    let _e171_ = phi_4193_;
                                    let _e173_ = phi_3853_;
                                    let _e175_ = phi_3856_;
                                    let _e177_ = phi_3858_;
                                    phi_4183_ = _e171_;
                                    phi_3852_ = _e173_;
                                    phi_3855_ = _e175_;
                                    phi_3857_ = _e177_;
                                    phi_4192_ = _e171_;
                                    if !(_e122_) {
                                        break;
                                    }
                                }
                                let _e180_ = phi_4192_;
                                phi_4205_ = _e180_;
                                if _e180_ {
                                    break;
                                }
                                let _e182_ = local_2_;
                                let _e183_ = bitcast<u32>(_e182_);
                                let _e184_ = (_e183_ & 2139095040u);
                                if (_e184_ == 2139095040u) {
                                    phi_3937_ = true;
                                } else {
                                    phi_3937_ = (_e184_ == 0u);
                                }
                                let _e188_ = phi_3937_;
                                let _e192_ = local_3_;
                                let _e193_ = bitcast<u32>(_e192_);
                                let _e194_ = (_e193_ & 2139095040u);
                                if (_e194_ == 2139095040u) {
                                    phi_3953_ = true;
                                } else {
                                    phi_3953_ = (_e194_ == 0u);
                                }
                                let _e198_ = phi_3953_;
                                let _e3880 = subgroupAdd(bitcast<f32>(select(_e183_, 0u, _e188_)));
                                let _e3884 = subgroupAdd(bitcast<f32>(select(_e193_, 0u, _e198_)));
                                let _e203_ = (_e57_ == 0u);
                                let _e207_ = select(255u, _e68_, (_e68_ < 255u));
                                let _e208_ = (_e207_ < 256u);
                                if _e208_ {
                                } else {
                                    phi_4205_ = true;
                                    break;
                                }
                                reduce_a[_e207_] = select(0f, _e3880, _e203_);
                                if _e208_ {
                                } else {
                                    phi_4205_ = true;
                                    break;
                                }
                                reduce_b[_e207_] = select(0f, _e3884, _e203_);
                                workgroupBarrier();
                                phi_3991_ = 128u;
                                loop {
                                    let _e212_ = phi_3991_;
                                    let _e213_ = (_e212_ > 0u);
                                    if _e213_ {
                                        if (_e68_ < _e212_) {
                                            let _e215_ = (_e68_ + _e212_);
                                            let _e216_ = (_e215_ < 256u);
                                            if _e216_ {
                                            } else {
                                                phi_4194_ = true;
                                                break;
                                            }
                                            let _e218_ = reduce_a[_e215_];
                                            let _e219_ = (_e68_ < 256u);
                                            if _e219_ {
                                            } else {
                                                phi_4194_ = true;
                                                break;
                                            }
                                            let _e221_ = reduce_a[_e68_];
                                            reduce_a[_e68_] = (_e221_ + _e218_);
                                            if _e216_ {
                                            } else {
                                                phi_4194_ = true;
                                                break;
                                            }
                                            let _e224_ = reduce_b[_e215_];
                                            if _e219_ {
                                            } else {
                                                phi_4194_ = true;
                                                break;
                                            }
                                            let _e226_ = reduce_b[_e68_];
                                            reduce_b[_e68_] = (_e226_ + _e224_);
                                        }
                                        workgroupBarrier();
                                        phi_3992_ = (_e212_ >> bitcast<u32>(1i));
                                    } else {
                                        phi_3992_ = u32();
                                    }
                                    let _e231_ = phi_3992_;
                                    phi_3991_ = _e231_;
                                    phi_4194_ = _e180_;
                                    if !(_e213_) {
                                        break;
                                    }
                                }
                                let _e234_ = phi_4194_;
                                phi_4205_ = _e234_;
                                if _e234_ {
                                    break;
                                }
                                let _e236_ = reduce_a[0];
                                let _e237_ = bitcast<u32>(_e236_);
                                let _e238_ = (_e237_ & 2139095040u);
                                if (_e238_ == 2139095040u) {
                                    phi_4052_ = true;
                                } else {
                                    phi_4052_ = (_e238_ == 0u);
                                }
                                let _e242_ = phi_4052_;
                                let _e246_ = reduce_b[0];
                                let _e247_ = bitcast<u32>(_e246_);
                                let _e248_ = (_e247_ & 2139095040u);
                                if (_e248_ == 2139095040u) {
                                    phi_4070_ = true;
                                } else {
                                    phi_4070_ = (_e248_ == 0u);
                                }
                                let _e252_ = phi_4070_;
                                workgroupBarrier();
                                if (_e68_ == 0u) {
                                    let _e256_ = bitcast<u32>(bitcast<f32>(select(_e237_, 0u, _e242_)));
                                    let _e257_ = (_e256_ & 2139095040u);
                                    if (_e257_ == 2139095040u) {
                                        phi_4090_ = true;
                                    } else {
                                        phi_4090_ = (_e257_ == 0u);
                                    }
                                    let _e261_ = phi_4090_;
                                    let _e264_ = (_e88_ + _e112_);
                                    let _e265_ = (_e65_ == 0u);
                                    if _e265_ {
                                        phi_4107_ = 0u;
                                    } else {
                                        let _e266_ = (_e65_ - 1u);
                                        phi_4107_ = select(_e266_, _e264_, (_e264_ < _e266_));
                                    }
                                    let _e270_ = phi_4107_;
                                    if (_e270_ < _e65_) {
                                    } else {
                                        phi_4205_ = true;
                                        break;
                                    }
                                    tile_weight_grads.member[_e270_] = bitcast<f32>(select(_e256_, 0u, _e261_));
                                    let _e274_ = bitcast<u32>(bitcast<f32>(select(_e247_, 0u, _e252_)));
                                    let _e275_ = (_e274_ & 2139095040u);
                                    if (_e275_ == 2139095040u) {
                                        phi_4124_ = true;
                                    } else {
                                        phi_4124_ = (_e275_ == 0u);
                                    }
                                    let _e279_ = phi_4124_;
                                    let _e282_ = (_e88_ + _e113_);
                                    if _e265_ {
                                        phi_4141_ = 0u;
                                    } else {
                                        let _e283_ = (_e65_ - 1u);
                                        phi_4141_ = select(_e283_, _e282_, (_e282_ < _e283_));
                                    }
                                    let _e287_ = phi_4141_;
                                    if (_e287_ < _e65_) {
                                    } else {
                                        phi_4205_ = true;
                                        break;
                                    }
                                    tile_weight_grads.member[_e287_] = bitcast<f32>(select(_e274_, 0u, _e279_));
                                }
                                phi_4206_ = _e234_;
                            } else {
                                phi_4206_ = _e90_;
                            }
                            let _e292_ = phi_4206_;
                            phi_4185_ = _e292_;
                            phi_3817_ = _e105_;
                            phi_4205_ = _e292_;
                            if !(_e111_) {
                                break;
                            }
                        }
                        let _e295_ = phi_4205_;
                        phi_4427_ = _e295_;
                        if _e295_ {
                            break;
                        }
                        phi_4428_ = _e295_;
                    }
                    let _e297_ = phi_4428_;
                    phi_4427_ = _e297_;
                    break;
                }
                default: {
                    phi_4427_ = false;
                    break;
                }
            }
            let _e2530_ = phi_4427_;
            if _e2530_ {
                break;
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn tile_mix_fwd_adj(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
