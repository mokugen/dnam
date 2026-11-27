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
    var phi_4127_: bool;
    var phi_3767_: type_19_;
    var phi_3768_: type_19_;
    var phi_3791_: type_19_;
    var phi_4125_: bool;
    var phi_3802_: u32;
    var phi_3805_: f32;
    var phi_3807_: f32;
    var phi_3816_: u32;
    var phi_3819_: f32;
    var phi_3821_: f32;
    var phi_3846_: u32;
    var phi_3866_: u32;
    var phi_3817_: u32;
    var phi_3820_: f32;
    var phi_3822_: f32;
    var phi_4122_: bool;
    var phi_4135_: bool;
    var phi_3803_: u32;
    var phi_3806_: f32;
    var phi_3808_: f32;
    var phi_4134_: bool;
    var local_2_: f32;
    var phi_3887_: bool;
    var local_3_: f32;
    var phi_3903_: bool;
    var phi_3933_: u32;
    var phi_3934_: u32;
    var phi_4136_: bool;
    var phi_3994_: bool;
    var phi_4012_: bool;
    var phi_4032_: bool;
    var phi_4049_: u32;
    var phi_4066_: bool;
    var phi_4083_: u32;
    var phi_4147_: bool;
    var phi_4146_: bool;
    var phi_4363_: bool;
    var phi_2302_: vec4<f32>;
    var phi_2342_: vec4<f32>;
    var phi_2355_: vec4<f32>;
    var phi_2368_: vec4<f32>;
    var phi_2381_: vec4<f32>;
    var phi_2394_: vec4<f32>;
    var phi_2407_: vec4<f32>;
    var phi_2419_: vec4<f32>;
    var phi_2432_: vec4<f32>;
    var phi_2441_: f32;
    var phi_2442_: f32;
    var phi_2443_: f32;
    var phi_2444_: f32;
    var phi_2445_: f32;
    var phi_2446_: f32;
    var phi_2447_: f32;
    var phi_2448_: f32;
    var phi_2449_: vec4<f32>;
    var phi_2450_: vec4<f32>;
    var phi_2451_: vec4<f32>;
    var phi_2452_: vec4<f32>;
    var phi_2453_: vec4<f32>;
    var phi_2454_: vec4<f32>;
    var phi_2456_: f32;
    var phi_2457_: f32;
    var phi_2458_: f32;
    var phi_2459_: f32;
    var phi_2460_: f32;
    var phi_2461_: f32;
    var phi_2462_: f32;
    var phi_2463_: f32;
    var phi_2464_: vec4<f32>;
    var phi_2465_: vec4<f32>;
    var phi_2466_: vec4<f32>;
    var phi_2467_: vec4<f32>;
    var phi_2468_: vec4<f32>;
    var phi_2469_: vec4<f32>;
    var phi_2471_: bool;
    var phi_2472_: f32;
    var phi_2473_: f32;
    var phi_2474_: f32;
    var phi_2475_: f32;
    var phi_2476_: f32;
    var phi_2477_: f32;
    var phi_2478_: f32;
    var phi_2479_: f32;
    var phi_2480_: vec4<f32>;
    var phi_2481_: vec4<f32>;
    var phi_2482_: vec4<f32>;
    var phi_2483_: vec4<f32>;
    var phi_2484_: vec4<f32>;
    var phi_2485_: vec4<f32>;
    var phi_2486_: bool;
    var phi_2487_: bool;
    var phi_4173_: bool;
    var phi_2525_: type_19_;
    var phi_2526_: type_19_;
    var phi_2549_: type_19_;
    var phi_2567_: f32;
    var phi_2578_: f32;
    var phi_2595_: bool;
    var phi_2611_: bool;
    var phi_2641_: u32;
    var phi_2642_: u32;
    var phi_4161_: bool;
    var phi_2702_: bool;
    var phi_2720_: bool;
    var phi_2742_: bool;
    var phi_2758_: bool;
    var phi_2788_: u32;
    var phi_2789_: u32;
    var phi_4191_: bool;
    var phi_2849_: bool;
    var phi_2867_: bool;
    var phi_2883_: f32;
    var phi_2894_: f32;
    var phi_2911_: bool;
    var phi_2927_: bool;
    var phi_2957_: u32;
    var phi_2958_: u32;
    var phi_4205_: bool;
    var phi_3018_: bool;
    var phi_3036_: bool;
    var phi_3058_: bool;
    var phi_3074_: bool;
    var phi_3104_: u32;
    var phi_3105_: u32;
    var phi_4221_: bool;
    var phi_3165_: bool;
    var phi_3183_: bool;
    var phi_3199_: f32;
    var phi_3210_: f32;
    var phi_3227_: bool;
    var phi_3243_: bool;
    var phi_3273_: u32;
    var phi_3274_: u32;
    var phi_4235_: bool;
    var phi_3334_: bool;
    var phi_3352_: bool;
    var phi_3374_: bool;
    var phi_3390_: bool;
    var phi_3420_: u32;
    var phi_3421_: u32;
    var phi_4251_: bool;
    var phi_3481_: bool;
    var phi_3499_: bool;
    var phi_3521_: u32;
    var phi_3540_: u32;
    var phi_3559_: u32;
    var phi_3578_: u32;
    var phi_3597_: u32;
    var phi_3617_: u32;
    var phi_3637_: u32;
    var phi_3657_: u32;
    var phi_3676_: u32;
    var phi_3696_: u32;
    var phi_3716_: u32;
    var phi_3736_: u32;
    var phi_4266_: bool;
    var phi_4265_: bool;
    var phi_4365_: bool;
    var phi_4364_: bool;
    var phi_1044_: u32;
    var phi_1045_: u32;
    var phi_1046_: u32;
    var phi_1075_: vec4<f32>;
    var phi_1088_: vec4<f32>;
    var phi_1101_: vec4<f32>;
    var phi_1114_: vec4<f32>;
    var phi_1127_: vec4<f32>;
    var phi_1140_: vec4<f32>;
    var phi_1143_: type_19_;
    var phi_1146_: vec4<f32>;
    var phi_1148_: vec4<f32>;
    var phi_1150_: vec4<f32>;
    var phi_1152_: vec4<f32>;
    var phi_1154_: vec4<f32>;
    var phi_1156_: vec4<f32>;
    var phi_1144_: type_19_;
    var phi_1179_: type_19_;
    var phi_1199_: vec4<f32>;
    var phi_1228_: vec4<f32>;
    var phi_1241_: vec4<f32>;
    var phi_1254_: vec4<f32>;
    var phi_1267_: vec4<f32>;
    var phi_1280_: vec4<f32>;
    var phi_1293_: vec4<f32>;
    var phi_1306_: vec4<f32>;
    var phi_1319_: vec4<f32>;
    var phi_1332_: vec4<f32>;
    var phi_1345_: vec4<f32>;
    var phi_1358_: vec4<f32>;
    var phi_1371_: vec4<f32>;
    var phi_1384_: vec4<f32>;
    var phi_1397_: vec4<f32>;
    var phi_1770_: vec4<f32>;
    var phi_1771_: vec4<f32>;
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
    var phi_1147_: vec4<f32>;
    var phi_1149_: vec4<f32>;
    var phi_1151_: vec4<f32>;
    var phi_1153_: vec4<f32>;
    var phi_1155_: vec4<f32>;
    var phi_1157_: vec4<f32>;
    var phi_4296_: bool;
    var local_4_: vec4<f32>;
    var phi_1793_: bool;
    var local_5_: vec4<f32>;
    var phi_1810_: bool;
    var local_6_: vec4<f32>;
    var phi_1827_: bool;
    var local_7_: vec4<f32>;
    var phi_1844_: bool;
    var local_8_: vec4<f32>;
    var phi_1873_: bool;
    var local_9_: vec4<f32>;
    var phi_1890_: bool;
    var local_10_: vec4<f32>;
    var phi_1907_: bool;
    var local_11_: vec4<f32>;
    var phi_1924_: bool;
    var local_12_: vec4<f32>;
    var phi_1953_: bool;
    var local_13_: vec4<f32>;
    var phi_1970_: bool;
    var local_14_: vec4<f32>;
    var phi_1987_: bool;
    var local_15_: vec4<f32>;
    var phi_2004_: bool;
    var local_16_: vec4<f32>;
    var phi_2033_: bool;
    var local_17_: vec4<f32>;
    var phi_2050_: bool;
    var local_18_: vec4<f32>;
    var phi_2067_: bool;
    var local_19_: vec4<f32>;
    var phi_2084_: bool;
    var local_20_: vec4<f32>;
    var phi_2113_: bool;
    var local_21_: vec4<f32>;
    var phi_2130_: bool;
    var local_22_: vec4<f32>;
    var phi_2147_: bool;
    var local_23_: vec4<f32>;
    var phi_2164_: bool;
    var local_24_: vec4<f32>;
    var phi_2193_: bool;
    var local_25_: vec4<f32>;
    var phi_2210_: bool;
    var local_26_: vec4<f32>;
    var phi_2227_: bool;
    var local_27_: vec4<f32>;
    var phi_2244_: bool;
    var phi_4367_: bool;
    var phi_4366_: bool;
    var phi_156_: u32;
    var phi_157_: u32;
    var phi_186_: vec4<f32>;
    var phi_199_: vec4<f32>;
    var phi_202_: type_19_;
    var phi_205_: vec4<f32>;
    var phi_207_: vec4<f32>;
    var phi_203_: type_19_;
    var phi_230_: type_19_;
    var phi_250_: vec4<f32>;
    var phi_279_: vec4<f32>;
    var phi_292_: vec4<f32>;
    var phi_305_: vec4<f32>;
    var phi_318_: vec4<f32>;
    var phi_331_: vec4<f32>;
    var phi_344_: vec4<f32>;
    var phi_357_: vec4<f32>;
    var phi_370_: vec4<f32>;
    var phi_383_: vec4<f32>;
    var phi_396_: vec4<f32>;
    var phi_409_: vec4<f32>;
    var phi_422_: vec4<f32>;
    var phi_435_: vec4<f32>;
    var phi_448_: vec4<f32>;
    var phi_461_: vec4<f32>;
    var phi_474_: vec4<f32>;
    var phi_487_: vec4<f32>;
    var phi_500_: vec4<f32>;
    var phi_823_: vec4<f32>;
    var phi_824_: vec4<f32>;
    var phi_825_: vec4<f32>;
    var phi_826_: vec4<f32>;
    var phi_206_: vec4<f32>;
    var phi_208_: vec4<f32>;
    var phi_4330_: bool;
    var local_28_: vec4<f32>;
    var phi_838_: bool;
    var local_29_: vec4<f32>;
    var phi_855_: bool;
    var local_30_: vec4<f32>;
    var phi_872_: bool;
    var local_31_: vec4<f32>;
    var phi_889_: bool;
    var local_32_: vec4<f32>;
    var phi_918_: bool;
    var local_33_: vec4<f32>;
    var phi_935_: bool;
    var local_34_: vec4<f32>;
    var phi_952_: bool;
    var local_35_: vec4<f32>;
    var phi_969_: bool;
    var phi_4423_: bool;
    var phi_4422_: bool;
    var phi_4362_: bool;
    var local_36_: f32;
    var local_37_: f32;

    switch bitcast<i32>(0u) {
        default: {
            let _e54_ = wg_id_1_;
            let _e55_ = lid_1_;
            let _e57_ = arrayLength((&block_inputs_or_d_loss_d_inputs.member));
            let _e59_ = arrayLength((&tile_descriptors.member));
            let _e61_ = arrayLength((&feature_outputs.member));
            let _e63_ = arrayLength((&tile_weight_grads.member));
            let _e65_ = arrayLength((&d_loss_d_features.member));
            let _e66_ = (_e55_ & 255u);
            let _e69_ = uniforms.member.member_3_;
            switch bitcast<i32>(_e69_) {
                case 0: {
                    let _e1890_ = uniforms.member.member;
                    let _e1893_ = uniforms.member.member_6_;
                    let _e1895_ = select(1u, _e1893_, (1u < _e1893_));
                    let _e1898_ = uniforms.member.member_2_;
                    let _e1901_ = uniforms.member.member_5_;
                    if (_e1901_ == 4294967295u) {
                        phi_157_ = select(16u, _e54_.x, (_e54_.x < _e1895_));
                    } else {
                        let _e1904_ = (_e54_.x < 4u);
                        if _e1904_ {
                            local_1_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 8u, 16u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 6u, 16u, 16u), array<u32, 4>(3u, 7u, 16u, 16u));
                            let _e1905_ = (_e1901_ & 3u);
                            if (_e1905_ < 4u) {
                            } else {
                                phi_4362_ = true;
                                break;
                            }
                            if _e1904_ {
                            } else {
                                phi_4362_ = true;
                                break;
                            }
                            let _e1909_ = local_1_[_e1905_][_e54_.x];
                            phi_156_ = select(16u, _e1909_, (_e1909_ < _e1895_));
                        } else {
                            phi_156_ = 16u;
                        }
                        let _e1913_ = phi_156_;
                        phi_157_ = _e1913_;
                    }
                    let _e1917_ = phi_157_;
                    if (_e1917_ >= _e1895_) {
                        phi_4422_ = false;
                    } else {
                        let _e1921_ = ((_e54_.y * 256u) + _e66_);
                        if (_e1921_ >= _e1890_) {
                            phi_4423_ = false;
                        } else {
                            let _e1924_ = (_e54_.z * _e1895_);
                            let _e1928_ = ((((_e1924_ + _e1917_) * _e1890_) + _e1921_) * 2u);
                            let _e1929_ = (_e1928_ < _e57_);
                            if _e1929_ {
                                if _e1929_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1932_ = block_inputs_or_d_loss_d_inputs.member[_e1928_];
                                phi_186_ = _e1932_;
                            } else {
                                phi_186_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1934_ = phi_186_;
                            let _e1935_ = (_e1928_ + 1u);
                            let _e1936_ = (_e1935_ < _e57_);
                            if _e1936_ {
                                if _e1936_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1939_ = block_inputs_or_d_loss_d_inputs.member[_e1935_];
                                phi_199_ = _e1939_;
                            } else {
                                phi_199_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1941_ = phi_199_;
                            phi_202_ = type_19_(0u, _e1898_);
                            phi_205_ = _e1941_;
                            phi_207_ = _e1934_;
                            loop {
                                let _e1944_ = phi_202_;
                                let _e1946_ = phi_205_;
                                let _e1948_ = phi_207_;
                                local_28_ = _e1948_;
                                local_29_ = _e1948_;
                                local_30_ = _e1948_;
                                local_31_ = _e1948_;
                                local_32_ = _e1946_;
                                local_33_ = _e1946_;
                                local_34_ = _e1946_;
                                local_35_ = _e1946_;
                                if (_e1944_.member < _e1944_.member_1_) {
                                    phi_203_ = type_19_((_e1944_.member + 1u), _e1944_.member_1_);
                                    phi_230_ = type_19_(1u, _e1944_.member);
                                } else {
                                    phi_203_ = _e1944_;
                                    phi_230_ = type_19_(0u, u32());
                                }
                                let _e1961_ = phi_203_;
                                let _e1963_ = phi_230_;
                                let _e1967_ = (bitcast<i32>(_e1963_.member) != 0i);
                                if _e1967_ {
                                    let _e1968_ = (_e1963_.member_1_ * 13u);
                                    let _e1969_ = (_e1968_ < _e59_);
                                    if _e1969_ {
                                        if _e1969_ {
                                        } else {
                                            phi_4330_ = true;
                                            break;
                                        }
                                        let _e1972_ = tile_descriptors.member[_e1968_];
                                        phi_250_ = _e1972_;
                                    } else {
                                        phi_250_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1974_ = phi_250_;
                                    let _e1976_ = bitcast<u32>(_e1974_.x);
                                    if (bitcast<u32>(_e1974_.y) == _e1917_) {
                                        if (_e1976_ < _e1895_) {
                                            let _e1984_ = ((((_e1924_ + _e1976_) * _e1890_) + _e1921_) * 6u);
                                            let _e1985_ = (_e1984_ < _e61_);
                                            if _e1985_ {
                                                if _e1985_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e1988_ = feature_outputs.member[_e1984_];
                                                phi_279_ = _e1988_;
                                            } else {
                                                phi_279_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1990_ = phi_279_;
                                            let _e1991_ = (_e1984_ + 1u);
                                            let _e1992_ = (_e1991_ < _e61_);
                                            if _e1992_ {
                                                if _e1992_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e1995_ = feature_outputs.member[_e1991_];
                                                phi_292_ = _e1995_;
                                            } else {
                                                phi_292_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1997_ = phi_292_;
                                            let _e1998_ = (_e1984_ + 2u);
                                            let _e1999_ = (_e1998_ < _e61_);
                                            if _e1999_ {
                                                if _e1999_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2002_ = feature_outputs.member[_e1998_];
                                                phi_305_ = _e2002_;
                                            } else {
                                                phi_305_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2004_ = phi_305_;
                                            let _e2005_ = (_e1984_ + 3u);
                                            let _e2006_ = (_e2005_ < _e61_);
                                            if _e2006_ {
                                                if _e2006_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2009_ = feature_outputs.member[_e2005_];
                                                phi_318_ = _e2009_;
                                            } else {
                                                phi_318_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2011_ = phi_318_;
                                            let _e2012_ = (_e1984_ + 4u);
                                            let _e2013_ = (_e2012_ < _e61_);
                                            if _e2013_ {
                                                if _e2013_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2016_ = feature_outputs.member[_e2012_];
                                                phi_331_ = _e2016_;
                                            } else {
                                                phi_331_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2018_ = phi_331_;
                                            let _e2019_ = (_e1984_ + 5u);
                                            let _e2020_ = (_e2019_ < _e61_);
                                            if _e2020_ {
                                                if _e2020_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2023_ = feature_outputs.member[_e2019_];
                                                phi_344_ = _e2023_;
                                            } else {
                                                phi_344_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2025_ = phi_344_;
                                            let _e2026_ = (_e1968_ + 1u);
                                            let _e2027_ = (_e2026_ < _e59_);
                                            if _e2027_ {
                                                if _e2027_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2030_ = tile_descriptors.member[_e2026_];
                                                phi_357_ = _e2030_;
                                            } else {
                                                phi_357_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2032_ = phi_357_;
                                            let _e2033_ = (_e1968_ + 2u);
                                            let _e2034_ = (_e2033_ < _e59_);
                                            if _e2034_ {
                                                if _e2034_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2037_ = tile_descriptors.member[_e2033_];
                                                phi_370_ = _e2037_;
                                            } else {
                                                phi_370_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2039_ = phi_370_;
                                            let _e2040_ = (_e1968_ + 3u);
                                            let _e2041_ = (_e2040_ < _e59_);
                                            if _e2041_ {
                                                if _e2041_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2044_ = tile_descriptors.member[_e2040_];
                                                phi_383_ = _e2044_;
                                            } else {
                                                phi_383_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2046_ = phi_383_;
                                            let _e2047_ = (_e1968_ + 4u);
                                            let _e2048_ = (_e2047_ < _e59_);
                                            if _e2048_ {
                                                if _e2048_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2051_ = tile_descriptors.member[_e2047_];
                                                phi_396_ = _e2051_;
                                            } else {
                                                phi_396_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2053_ = phi_396_;
                                            let _e2054_ = (_e1968_ + 5u);
                                            let _e2055_ = (_e2054_ < _e59_);
                                            if _e2055_ {
                                                if _e2055_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2058_ = tile_descriptors.member[_e2054_];
                                                phi_409_ = _e2058_;
                                            } else {
                                                phi_409_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2060_ = phi_409_;
                                            let _e2061_ = (_e1968_ + 6u);
                                            let _e2062_ = (_e2061_ < _e59_);
                                            if _e2062_ {
                                                if _e2062_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2065_ = tile_descriptors.member[_e2061_];
                                                phi_422_ = _e2065_;
                                            } else {
                                                phi_422_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2067_ = phi_422_;
                                            let _e2068_ = (_e1968_ + 7u);
                                            let _e2069_ = (_e2068_ < _e59_);
                                            if _e2069_ {
                                                if _e2069_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2072_ = tile_descriptors.member[_e2068_];
                                                phi_435_ = _e2072_;
                                            } else {
                                                phi_435_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2074_ = phi_435_;
                                            let _e2075_ = (_e1968_ + 8u);
                                            let _e2076_ = (_e2075_ < _e59_);
                                            if _e2076_ {
                                                if _e2076_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2079_ = tile_descriptors.member[_e2075_];
                                                phi_448_ = _e2079_;
                                            } else {
                                                phi_448_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2081_ = phi_448_;
                                            let _e2082_ = (_e1968_ + 9u);
                                            let _e2083_ = (_e2082_ < _e59_);
                                            if _e2083_ {
                                                if _e2083_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2086_ = tile_descriptors.member[_e2082_];
                                                phi_461_ = _e2086_;
                                            } else {
                                                phi_461_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2088_ = phi_461_;
                                            let _e2089_ = (_e1968_ + 10u);
                                            let _e2090_ = (_e2089_ < _e59_);
                                            if _e2090_ {
                                                if _e2090_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2093_ = tile_descriptors.member[_e2089_];
                                                phi_474_ = _e2093_;
                                            } else {
                                                phi_474_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2095_ = phi_474_;
                                            let _e2096_ = (_e1968_ + 11u);
                                            let _e2097_ = (_e2096_ < _e59_);
                                            if _e2097_ {
                                                if _e2097_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2100_ = tile_descriptors.member[_e2096_];
                                                phi_487_ = _e2100_;
                                            } else {
                                                phi_487_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2102_ = phi_487_;
                                            let _e2103_ = (_e1968_ + 12u);
                                            let _e2104_ = (_e2103_ < _e59_);
                                            if _e2104_ {
                                                if _e2104_ {
                                                } else {
                                                    phi_4330_ = true;
                                                    break;
                                                }
                                                let _e2107_ = tile_descriptors.member[_e2103_];
                                                phi_500_ = _e2107_;
                                            } else {
                                                phi_500_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e2109_ = phi_500_;
                                            phi_823_ = vec4<f32>(((((((((((((_e1946_.x + (_e2032_.x * _e2011_.x)) + (_e2039_.x * _e2011_.y)) + (_e2046_.x * _e2011_.z)) + (_e2053_.x * _e2011_.w)) + (_e2060_.x * _e2025_.x)) + (_e2067_.x * _e2025_.y)) + (_e2074_.x * _e2025_.z)) + (_e2081_.x * _e2025_.w)) + (_e2088_.x * _e1997_.x)) + (_e2095_.x * _e1997_.y)) + (_e2102_.x * _e1997_.z)) + (_e2109_.x * _e1997_.w)), ((((((((((((_e1946_.y + (_e2032_.y * _e2011_.x)) + (_e2039_.y * _e2011_.y)) + (_e2046_.y * _e2011_.z)) + (_e2053_.y * _e2011_.w)) + (_e2060_.y * _e2025_.x)) + (_e2067_.y * _e2025_.y)) + (_e2074_.y * _e2025_.z)) + (_e2081_.y * _e2025_.w)) + (_e2088_.y * _e1997_.x)) + (_e2095_.y * _e1997_.y)) + (_e2102_.y * _e1997_.z)) + (_e2109_.y * _e1997_.w)), ((((((((((((_e1946_.z + (_e2032_.z * _e2011_.x)) + (_e2039_.z * _e2011_.y)) + (_e2046_.z * _e2011_.z)) + (_e2053_.z * _e2011_.w)) + (_e2060_.z * _e2025_.x)) + (_e2067_.z * _e2025_.y)) + (_e2074_.z * _e2025_.z)) + (_e2081_.z * _e2025_.w)) + (_e2088_.z * _e1997_.x)) + (_e2095_.z * _e1997_.y)) + (_e2102_.z * _e1997_.z)) + (_e2109_.z * _e1997_.w)), ((((((((((((_e1946_.w + (_e2032_.w * _e2011_.x)) + (_e2039_.w * _e2011_.y)) + (_e2046_.w * _e2011_.z)) + (_e2053_.w * _e2011_.w)) + (_e2060_.w * _e2025_.x)) + (_e2067_.w * _e2025_.y)) + (_e2074_.w * _e2025_.z)) + (_e2081_.w * _e2025_.w)) + (_e2088_.w * _e1997_.x)) + (_e2095_.w * _e1997_.y)) + (_e2102_.w * _e1997_.z)) + (_e2109_.w * _e1997_.w)));
                                            phi_824_ = vec4<f32>(((((((((((((_e1948_.x + (_e2032_.x * _e2004_.x)) + (_e2039_.x * _e2004_.y)) + (_e2046_.x * _e2004_.z)) + (_e2053_.x * _e2004_.w)) + (_e2060_.x * _e2018_.x)) + (_e2067_.x * _e2018_.y)) + (_e2074_.x * _e2018_.z)) + (_e2081_.x * _e2018_.w)) + (_e2088_.x * _e1990_.x)) + (_e2095_.x * _e1990_.y)) + (_e2102_.x * _e1990_.z)) + (_e2109_.x * _e1990_.w)), ((((((((((((_e1948_.y + (_e2032_.y * _e2004_.x)) + (_e2039_.y * _e2004_.y)) + (_e2046_.y * _e2004_.z)) + (_e2053_.y * _e2004_.w)) + (_e2060_.y * _e2018_.x)) + (_e2067_.y * _e2018_.y)) + (_e2074_.y * _e2018_.z)) + (_e2081_.y * _e2018_.w)) + (_e2088_.y * _e1990_.x)) + (_e2095_.y * _e1990_.y)) + (_e2102_.y * _e1990_.z)) + (_e2109_.y * _e1990_.w)), ((((((((((((_e1948_.z + (_e2032_.z * _e2004_.x)) + (_e2039_.z * _e2004_.y)) + (_e2046_.z * _e2004_.z)) + (_e2053_.z * _e2004_.w)) + (_e2060_.z * _e2018_.x)) + (_e2067_.z * _e2018_.y)) + (_e2074_.z * _e2018_.z)) + (_e2081_.z * _e2018_.w)) + (_e2088_.z * _e1990_.x)) + (_e2095_.z * _e1990_.y)) + (_e2102_.z * _e1990_.z)) + (_e2109_.z * _e1990_.w)), ((((((((((((_e1948_.w + (_e2032_.w * _e2004_.x)) + (_e2039_.w * _e2004_.y)) + (_e2046_.w * _e2004_.z)) + (_e2053_.w * _e2004_.w)) + (_e2060_.w * _e2018_.x)) + (_e2067_.w * _e2018_.y)) + (_e2074_.w * _e2018_.z)) + (_e2081_.w * _e2018_.w)) + (_e2088_.w * _e1990_.x)) + (_e2095_.w * _e1990_.y)) + (_e2102_.w * _e1990_.z)) + (_e2109_.w * _e1990_.w)));
                                        } else {
                                            phi_823_ = _e1946_;
                                            phi_824_ = _e1948_;
                                        }
                                        let _e2385_ = phi_823_;
                                        let _e2387_ = phi_824_;
                                        phi_825_ = _e2385_;
                                        phi_826_ = _e2387_;
                                    } else {
                                        phi_825_ = _e1946_;
                                        phi_826_ = _e1948_;
                                    }
                                    let _e2389_ = phi_825_;
                                    let _e2391_ = phi_826_;
                                    phi_206_ = _e2389_;
                                    phi_208_ = _e2391_;
                                } else {
                                    phi_206_ = vec4<f32>();
                                    phi_208_ = vec4<f32>();
                                }
                                let _e2393_ = phi_206_;
                                let _e2395_ = phi_208_;
                                phi_202_ = _e1961_;
                                phi_205_ = _e2393_;
                                phi_207_ = _e2395_;
                                phi_4330_ = false;
                                if !(_e1967_) {
                                    break;
                                }
                            }
                            let _e2398_ = phi_4330_;
                            phi_4362_ = _e2398_;
                            if _e2398_ {
                                break;
                            }
                            let _e2400_ = local_28_;
                            let _e2402_ = bitcast<u32>(_e2400_.x);
                            let _e2403_ = (_e2402_ & 2139095040u);
                            if (_e2403_ == 2139095040u) {
                                phi_838_ = true;
                            } else {
                                phi_838_ = (_e2403_ == 0u);
                            }
                            let _e2407_ = phi_838_;
                            let _e2411_ = local_29_;
                            let _e2413_ = bitcast<u32>(_e2411_.y);
                            let _e2414_ = (_e2413_ & 2139095040u);
                            if (_e2414_ == 2139095040u) {
                                phi_855_ = true;
                            } else {
                                phi_855_ = (_e2414_ == 0u);
                            }
                            let _e2418_ = phi_855_;
                            let _e2422_ = local_30_;
                            let _e2424_ = bitcast<u32>(_e2422_.z);
                            let _e2425_ = (_e2424_ & 2139095040u);
                            if (_e2425_ == 2139095040u) {
                                phi_872_ = true;
                            } else {
                                phi_872_ = (_e2425_ == 0u);
                            }
                            let _e2429_ = phi_872_;
                            let _e2433_ = local_31_;
                            let _e2435_ = bitcast<u32>(_e2433_.w);
                            let _e2436_ = (_e2435_ & 2139095040u);
                            if (_e2436_ == 2139095040u) {
                                phi_889_ = true;
                            } else {
                                phi_889_ = (_e2436_ == 0u);
                            }
                            let _e2440_ = phi_889_;
                            if _e1929_ {
                                if _e1929_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                block_inputs_or_d_loss_d_inputs.member[_e1928_] = vec4<f32>(bitcast<f32>(select(_e2402_, 0u, _e2407_)), bitcast<f32>(select(_e2413_, 0u, _e2418_)), bitcast<f32>(select(_e2424_, 0u, _e2429_)), bitcast<f32>(select(_e2435_, 0u, _e2440_)));
                            }
                            let _e2447_ = local_32_;
                            let _e2449_ = bitcast<u32>(_e2447_.x);
                            let _e2450_ = (_e2449_ & 2139095040u);
                            if (_e2450_ == 2139095040u) {
                                phi_918_ = true;
                            } else {
                                phi_918_ = (_e2450_ == 0u);
                            }
                            let _e2454_ = phi_918_;
                            let _e2458_ = local_33_;
                            let _e2460_ = bitcast<u32>(_e2458_.y);
                            let _e2461_ = (_e2460_ & 2139095040u);
                            if (_e2461_ == 2139095040u) {
                                phi_935_ = true;
                            } else {
                                phi_935_ = (_e2461_ == 0u);
                            }
                            let _e2465_ = phi_935_;
                            let _e2469_ = local_34_;
                            let _e2471_ = bitcast<u32>(_e2469_.z);
                            let _e2472_ = (_e2471_ & 2139095040u);
                            if (_e2472_ == 2139095040u) {
                                phi_952_ = true;
                            } else {
                                phi_952_ = (_e2472_ == 0u);
                            }
                            let _e2476_ = phi_952_;
                            let _e2480_ = local_35_;
                            let _e2482_ = bitcast<u32>(_e2480_.w);
                            let _e2483_ = (_e2482_ & 2139095040u);
                            if (_e2483_ == 2139095040u) {
                                phi_969_ = true;
                            } else {
                                phi_969_ = (_e2483_ == 0u);
                            }
                            let _e2487_ = phi_969_;
                            if _e1936_ {
                                if _e1936_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                block_inputs_or_d_loss_d_inputs.member[_e1935_] = vec4<f32>(bitcast<f32>(select(_e2449_, 0u, _e2454_)), bitcast<f32>(select(_e2460_, 0u, _e2465_)), bitcast<f32>(select(_e2471_, 0u, _e2476_)), bitcast<f32>(select(_e2482_, 0u, _e2487_)));
                            }
                            phi_4423_ = _e2398_;
                        }
                        let _e2494_ = phi_4423_;
                        phi_4422_ = _e2494_;
                    }
                    let _e2496_ = phi_4422_;
                    phi_4362_ = _e2496_;
                    break;
                }
                case 1: {
                    let _e1094_ = uniforms.member.member;
                    let _e1097_ = uniforms.member.member_6_;
                    let _e1099_ = select(1u, _e1097_, (1u < _e1097_));
                    let _e1102_ = uniforms.member.member_2_;
                    let _e1105_ = uniforms.member.member_5_;
                    let _e1106_ = (_e1105_ == 4294967295u);
                    if _e1106_ {
                        phi_1046_ = select(4294967295u, _e54_.x, (_e54_.x < _e1099_));
                    } else {
                        if _e1106_ {
                            phi_1045_ = select(16u, _e54_.x, (_e54_.x < _e1099_));
                        } else {
                            let _e1108_ = (_e54_.x < 4u);
                            if _e1108_ {
                                local = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 8u, 16u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 6u, 16u, 16u), array<u32, 4>(3u, 7u, 16u, 16u));
                                let _e1109_ = (_e1105_ & 3u);
                                if (_e1109_ < 4u) {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                if _e1108_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1113_ = local[_e1109_][_e54_.x];
                                phi_1044_ = select(16u, _e1113_, (_e1113_ < _e1099_));
                            } else {
                                phi_1044_ = 16u;
                            }
                            let _e1117_ = phi_1044_;
                            phi_1045_ = _e1117_;
                        }
                        let _e1121_ = phi_1045_;
                        phi_1046_ = _e1121_;
                    }
                    let _e1126_ = phi_1046_;
                    if (_e1126_ >= _e1099_) {
                        phi_4366_ = false;
                    } else {
                        let _e1130_ = ((_e54_.y * 256u) + _e66_);
                        if (_e1130_ >= _e1094_) {
                            phi_4367_ = false;
                        } else {
                            let _e1133_ = (_e54_.z * _e1099_);
                            let _e1137_ = ((((_e1133_ + _e1126_) * _e1094_) + _e1130_) * 6u);
                            let _e1138_ = (_e1137_ < _e65_);
                            if _e1138_ {
                                if _e1138_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1141_ = d_loss_d_features.member[_e1137_];
                                phi_1075_ = _e1141_;
                            } else {
                                phi_1075_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1143_ = phi_1075_;
                            let _e1144_ = (_e1137_ + 1u);
                            let _e1145_ = (_e1144_ < _e65_);
                            if _e1145_ {
                                if _e1145_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1148_ = d_loss_d_features.member[_e1144_];
                                phi_1088_ = _e1148_;
                            } else {
                                phi_1088_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1150_ = phi_1088_;
                            let _e1151_ = (_e1137_ + 2u);
                            let _e1152_ = (_e1151_ < _e65_);
                            if _e1152_ {
                                if _e1152_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1155_ = d_loss_d_features.member[_e1151_];
                                phi_1101_ = _e1155_;
                            } else {
                                phi_1101_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1157_ = phi_1101_;
                            let _e1158_ = (_e1137_ + 3u);
                            let _e1159_ = (_e1158_ < _e65_);
                            if _e1159_ {
                                if _e1159_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1162_ = d_loss_d_features.member[_e1158_];
                                phi_1114_ = _e1162_;
                            } else {
                                phi_1114_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1164_ = phi_1114_;
                            let _e1165_ = (_e1137_ + 4u);
                            let _e1166_ = (_e1165_ < _e65_);
                            if _e1166_ {
                                if _e1166_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1169_ = d_loss_d_features.member[_e1165_];
                                phi_1127_ = _e1169_;
                            } else {
                                phi_1127_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1171_ = phi_1127_;
                            let _e1172_ = (_e1137_ + 5u);
                            let _e1173_ = (_e1172_ < _e65_);
                            if _e1173_ {
                                if _e1173_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e1176_ = d_loss_d_features.member[_e1172_];
                                phi_1140_ = _e1176_;
                            } else {
                                phi_1140_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e1178_ = phi_1140_;
                            phi_1143_ = type_19_(0u, _e1102_);
                            phi_1146_ = _e1178_;
                            phi_1148_ = _e1171_;
                            phi_1150_ = _e1164_;
                            phi_1152_ = _e1157_;
                            phi_1154_ = _e1150_;
                            phi_1156_ = _e1143_;
                            loop {
                                let _e1181_ = phi_1143_;
                                let _e1183_ = phi_1146_;
                                let _e1185_ = phi_1148_;
                                let _e1187_ = phi_1150_;
                                let _e1189_ = phi_1152_;
                                let _e1191_ = phi_1154_;
                                let _e1193_ = phi_1156_;
                                local_4_ = _e1193_;
                                local_5_ = _e1193_;
                                local_6_ = _e1193_;
                                local_7_ = _e1193_;
                                local_8_ = _e1191_;
                                local_9_ = _e1191_;
                                local_10_ = _e1191_;
                                local_11_ = _e1191_;
                                local_12_ = _e1189_;
                                local_13_ = _e1189_;
                                local_14_ = _e1189_;
                                local_15_ = _e1189_;
                                local_16_ = _e1187_;
                                local_17_ = _e1187_;
                                local_18_ = _e1187_;
                                local_19_ = _e1187_;
                                local_20_ = _e1185_;
                                local_21_ = _e1185_;
                                local_22_ = _e1185_;
                                local_23_ = _e1185_;
                                local_24_ = _e1183_;
                                local_25_ = _e1183_;
                                local_26_ = _e1183_;
                                local_27_ = _e1183_;
                                if (_e1181_.member < _e1181_.member_1_) {
                                    phi_1144_ = type_19_((_e1181_.member + 1u), _e1181_.member_1_);
                                    phi_1179_ = type_19_(1u, _e1181_.member);
                                } else {
                                    phi_1144_ = _e1181_;
                                    phi_1179_ = type_19_(0u, u32());
                                }
                                let _e1206_ = phi_1144_;
                                let _e1208_ = phi_1179_;
                                let _e1212_ = (bitcast<i32>(_e1208_.member) != 0i);
                                if _e1212_ {
                                    let _e1213_ = (_e1208_.member_1_ * 13u);
                                    let _e1214_ = (_e1213_ < _e59_);
                                    if _e1214_ {
                                        if _e1214_ {
                                        } else {
                                            phi_4296_ = true;
                                            break;
                                        }
                                        let _e1217_ = tile_descriptors.member[_e1213_];
                                        phi_1199_ = _e1217_;
                                    } else {
                                        phi_1199_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e1219_ = phi_1199_;
                                    let _e1223_ = bitcast<u32>(_e1219_.y);
                                    if (bitcast<u32>(_e1219_.x) == _e1126_) {
                                        if (_e1223_ < _e1099_) {
                                            let _e1229_ = ((((_e1133_ + _e1223_) * _e1094_) + _e1130_) * 2u);
                                            let _e1230_ = (_e1229_ < _e57_);
                                            if _e1230_ {
                                                if _e1230_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1233_ = block_inputs_or_d_loss_d_inputs.member[_e1229_];
                                                phi_1228_ = _e1233_;
                                            } else {
                                                phi_1228_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1235_ = phi_1228_;
                                            let _e1236_ = (_e1229_ + 1u);
                                            let _e1237_ = (_e1236_ < _e57_);
                                            if _e1237_ {
                                                if _e1237_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1240_ = block_inputs_or_d_loss_d_inputs.member[_e1236_];
                                                phi_1241_ = _e1240_;
                                            } else {
                                                phi_1241_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1242_ = phi_1241_;
                                            let _e1243_ = (_e1213_ + 1u);
                                            let _e1244_ = (_e1243_ < _e59_);
                                            if _e1244_ {
                                                if _e1244_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1247_ = tile_descriptors.member[_e1243_];
                                                phi_1254_ = _e1247_;
                                            } else {
                                                phi_1254_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1249_ = phi_1254_;
                                            let _e1250_ = (_e1213_ + 2u);
                                            let _e1251_ = (_e1250_ < _e59_);
                                            if _e1251_ {
                                                if _e1251_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1254_ = tile_descriptors.member[_e1250_];
                                                phi_1267_ = _e1254_;
                                            } else {
                                                phi_1267_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1256_ = phi_1267_;
                                            let _e1257_ = (_e1213_ + 3u);
                                            let _e1258_ = (_e1257_ < _e59_);
                                            if _e1258_ {
                                                if _e1258_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1261_ = tile_descriptors.member[_e1257_];
                                                phi_1280_ = _e1261_;
                                            } else {
                                                phi_1280_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1263_ = phi_1280_;
                                            let _e1264_ = (_e1213_ + 4u);
                                            let _e1265_ = (_e1264_ < _e59_);
                                            if _e1265_ {
                                                if _e1265_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1268_ = tile_descriptors.member[_e1264_];
                                                phi_1293_ = _e1268_;
                                            } else {
                                                phi_1293_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1270_ = phi_1293_;
                                            let _e1271_ = (_e1213_ + 5u);
                                            let _e1272_ = (_e1271_ < _e59_);
                                            if _e1272_ {
                                                if _e1272_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1275_ = tile_descriptors.member[_e1271_];
                                                phi_1306_ = _e1275_;
                                            } else {
                                                phi_1306_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1277_ = phi_1306_;
                                            let _e1278_ = (_e1213_ + 6u);
                                            let _e1279_ = (_e1278_ < _e59_);
                                            if _e1279_ {
                                                if _e1279_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1282_ = tile_descriptors.member[_e1278_];
                                                phi_1319_ = _e1282_;
                                            } else {
                                                phi_1319_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1284_ = phi_1319_;
                                            let _e1285_ = (_e1213_ + 7u);
                                            let _e1286_ = (_e1285_ < _e59_);
                                            if _e1286_ {
                                                if _e1286_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1289_ = tile_descriptors.member[_e1285_];
                                                phi_1332_ = _e1289_;
                                            } else {
                                                phi_1332_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1291_ = phi_1332_;
                                            let _e1292_ = (_e1213_ + 8u);
                                            let _e1293_ = (_e1292_ < _e59_);
                                            if _e1293_ {
                                                if _e1293_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1296_ = tile_descriptors.member[_e1292_];
                                                phi_1345_ = _e1296_;
                                            } else {
                                                phi_1345_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1298_ = phi_1345_;
                                            let _e1299_ = (_e1213_ + 9u);
                                            let _e1300_ = (_e1299_ < _e59_);
                                            if _e1300_ {
                                                if _e1300_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1303_ = tile_descriptors.member[_e1299_];
                                                phi_1358_ = _e1303_;
                                            } else {
                                                phi_1358_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1305_ = phi_1358_;
                                            let _e1306_ = (_e1213_ + 10u);
                                            let _e1307_ = (_e1306_ < _e59_);
                                            if _e1307_ {
                                                if _e1307_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1310_ = tile_descriptors.member[_e1306_];
                                                phi_1371_ = _e1310_;
                                            } else {
                                                phi_1371_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1312_ = phi_1371_;
                                            let _e1313_ = (_e1213_ + 11u);
                                            let _e1314_ = (_e1313_ < _e59_);
                                            if _e1314_ {
                                                if _e1314_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1317_ = tile_descriptors.member[_e1313_];
                                                phi_1384_ = _e1317_;
                                            } else {
                                                phi_1384_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1319_ = phi_1384_;
                                            let _e1320_ = (_e1213_ + 12u);
                                            let _e1321_ = (_e1320_ < _e59_);
                                            if _e1321_ {
                                                if _e1321_ {
                                                } else {
                                                    phi_4296_ = true;
                                                    break;
                                                }
                                                let _e1324_ = tile_descriptors.member[_e1320_];
                                                phi_1397_ = _e1324_;
                                            } else {
                                                phi_1397_ = vec4<f32>(0f, 0f, 0f, 0f);
                                            }
                                            let _e1326_ = phi_1397_;
                                            phi_1770_ = (_e1183_ + vec4<f32>(((((_e1277_.x * _e1242_.x) + (_e1277_.y * _e1242_.y)) + (_e1277_.z * _e1242_.z)) + (_e1277_.w * _e1242_.w)), ((((_e1284_.x * _e1242_.x) + (_e1284_.y * _e1242_.y)) + (_e1284_.z * _e1242_.z)) + (_e1284_.w * _e1242_.w)), ((((_e1291_.x * _e1242_.x) + (_e1291_.y * _e1242_.y)) + (_e1291_.z * _e1242_.z)) + (_e1291_.w * _e1242_.w)), ((((_e1298_.x * _e1242_.x) + (_e1298_.y * _e1242_.y)) + (_e1298_.z * _e1242_.z)) + (_e1298_.w * _e1242_.w))));
                                            phi_1771_ = (_e1185_ + vec4<f32>(((((_e1277_.x * _e1235_.x) + (_e1277_.y * _e1235_.y)) + (_e1277_.z * _e1235_.z)) + (_e1277_.w * _e1235_.w)), ((((_e1284_.x * _e1235_.x) + (_e1284_.y * _e1235_.y)) + (_e1284_.z * _e1235_.z)) + (_e1284_.w * _e1235_.w)), ((((_e1291_.x * _e1235_.x) + (_e1291_.y * _e1235_.y)) + (_e1291_.z * _e1235_.z)) + (_e1291_.w * _e1235_.w)), ((((_e1298_.x * _e1235_.x) + (_e1298_.y * _e1235_.y)) + (_e1298_.z * _e1235_.z)) + (_e1298_.w * _e1235_.w))));
                                            phi_1772_ = (_e1187_ + vec4<f32>(((((_e1249_.x * _e1242_.x) + (_e1249_.y * _e1242_.y)) + (_e1249_.z * _e1242_.z)) + (_e1249_.w * _e1242_.w)), ((((_e1256_.x * _e1242_.x) + (_e1256_.y * _e1242_.y)) + (_e1256_.z * _e1242_.z)) + (_e1256_.w * _e1242_.w)), ((((_e1263_.x * _e1242_.x) + (_e1263_.y * _e1242_.y)) + (_e1263_.z * _e1242_.z)) + (_e1263_.w * _e1242_.w)), ((((_e1270_.x * _e1242_.x) + (_e1270_.y * _e1242_.y)) + (_e1270_.z * _e1242_.z)) + (_e1270_.w * _e1242_.w))));
                                            phi_1773_ = (_e1189_ + vec4<f32>(((((_e1249_.x * _e1235_.x) + (_e1249_.y * _e1235_.y)) + (_e1249_.z * _e1235_.z)) + (_e1249_.w * _e1235_.w)), ((((_e1256_.x * _e1235_.x) + (_e1256_.y * _e1235_.y)) + (_e1256_.z * _e1235_.z)) + (_e1256_.w * _e1235_.w)), ((((_e1263_.x * _e1235_.x) + (_e1263_.y * _e1235_.y)) + (_e1263_.z * _e1235_.z)) + (_e1263_.w * _e1235_.w)), ((((_e1270_.x * _e1235_.x) + (_e1270_.y * _e1235_.y)) + (_e1270_.z * _e1235_.z)) + (_e1270_.w * _e1235_.w))));
                                            phi_1774_ = (_e1191_ + vec4<f32>(((((_e1305_.x * _e1242_.x) + (_e1305_.y * _e1242_.y)) + (_e1305_.z * _e1242_.z)) + (_e1305_.w * _e1242_.w)), ((((_e1312_.x * _e1242_.x) + (_e1312_.y * _e1242_.y)) + (_e1312_.z * _e1242_.z)) + (_e1312_.w * _e1242_.w)), ((((_e1319_.x * _e1242_.x) + (_e1319_.y * _e1242_.y)) + (_e1319_.z * _e1242_.z)) + (_e1319_.w * _e1242_.w)), ((((_e1326_.x * _e1242_.x) + (_e1326_.y * _e1242_.y)) + (_e1326_.z * _e1242_.z)) + (_e1326_.w * _e1242_.w))));
                                            phi_1775_ = (_e1193_ + vec4<f32>(((((_e1305_.x * _e1235_.x) + (_e1305_.y * _e1235_.y)) + (_e1305_.z * _e1235_.z)) + (_e1305_.w * _e1235_.w)), ((((_e1312_.x * _e1235_.x) + (_e1312_.y * _e1235_.y)) + (_e1312_.z * _e1235_.z)) + (_e1312_.w * _e1235_.w)), ((((_e1319_.x * _e1235_.x) + (_e1319_.y * _e1235_.y)) + (_e1319_.z * _e1235_.z)) + (_e1319_.w * _e1235_.w)), ((((_e1326_.x * _e1235_.x) + (_e1326_.y * _e1235_.y)) + (_e1326_.z * _e1235_.z)) + (_e1326_.w * _e1235_.w))));
                                        } else {
                                            phi_1770_ = _e1183_;
                                            phi_1771_ = _e1185_;
                                            phi_1772_ = _e1187_;
                                            phi_1773_ = _e1189_;
                                            phi_1774_ = _e1191_;
                                            phi_1775_ = _e1193_;
                                        }
                                        let _e1564_ = phi_1770_;
                                        let _e1566_ = phi_1771_;
                                        let _e1568_ = phi_1772_;
                                        let _e1570_ = phi_1773_;
                                        let _e1572_ = phi_1774_;
                                        let _e1574_ = phi_1775_;
                                        phi_1776_ = _e1564_;
                                        phi_1777_ = _e1566_;
                                        phi_1778_ = _e1568_;
                                        phi_1779_ = _e1570_;
                                        phi_1780_ = _e1572_;
                                        phi_1781_ = _e1574_;
                                    } else {
                                        phi_1776_ = _e1183_;
                                        phi_1777_ = _e1185_;
                                        phi_1778_ = _e1187_;
                                        phi_1779_ = _e1189_;
                                        phi_1780_ = _e1191_;
                                        phi_1781_ = _e1193_;
                                    }
                                    let _e1576_ = phi_1776_;
                                    let _e1578_ = phi_1777_;
                                    let _e1580_ = phi_1778_;
                                    let _e1582_ = phi_1779_;
                                    let _e1584_ = phi_1780_;
                                    let _e1586_ = phi_1781_;
                                    phi_1147_ = _e1576_;
                                    phi_1149_ = _e1578_;
                                    phi_1151_ = _e1580_;
                                    phi_1153_ = _e1582_;
                                    phi_1155_ = _e1584_;
                                    phi_1157_ = _e1586_;
                                } else {
                                    phi_1147_ = vec4<f32>();
                                    phi_1149_ = vec4<f32>();
                                    phi_1151_ = vec4<f32>();
                                    phi_1153_ = vec4<f32>();
                                    phi_1155_ = vec4<f32>();
                                    phi_1157_ = vec4<f32>();
                                }
                                let _e1588_ = phi_1147_;
                                let _e1590_ = phi_1149_;
                                let _e1592_ = phi_1151_;
                                let _e1594_ = phi_1153_;
                                let _e1596_ = phi_1155_;
                                let _e1598_ = phi_1157_;
                                phi_1143_ = _e1206_;
                                phi_1146_ = _e1588_;
                                phi_1148_ = _e1590_;
                                phi_1150_ = _e1592_;
                                phi_1152_ = _e1594_;
                                phi_1154_ = _e1596_;
                                phi_1156_ = _e1598_;
                                phi_4296_ = false;
                                if !(_e1212_) {
                                    break;
                                }
                            }
                            let _e1601_ = phi_4296_;
                            phi_4362_ = _e1601_;
                            if _e1601_ {
                                break;
                            }
                            let _e1603_ = local_4_;
                            let _e1605_ = bitcast<u32>(_e1603_.x);
                            let _e1606_ = (_e1605_ & 2139095040u);
                            if (_e1606_ == 2139095040u) {
                                phi_1793_ = true;
                            } else {
                                phi_1793_ = (_e1606_ == 0u);
                            }
                            let _e1610_ = phi_1793_;
                            let _e1614_ = local_5_;
                            let _e1616_ = bitcast<u32>(_e1614_.y);
                            let _e1617_ = (_e1616_ & 2139095040u);
                            if (_e1617_ == 2139095040u) {
                                phi_1810_ = true;
                            } else {
                                phi_1810_ = (_e1617_ == 0u);
                            }
                            let _e1621_ = phi_1810_;
                            let _e1625_ = local_6_;
                            let _e1627_ = bitcast<u32>(_e1625_.z);
                            let _e1628_ = (_e1627_ & 2139095040u);
                            if (_e1628_ == 2139095040u) {
                                phi_1827_ = true;
                            } else {
                                phi_1827_ = (_e1628_ == 0u);
                            }
                            let _e1632_ = phi_1827_;
                            let _e1636_ = local_7_;
                            let _e1638_ = bitcast<u32>(_e1636_.w);
                            let _e1639_ = (_e1638_ & 2139095040u);
                            if (_e1639_ == 2139095040u) {
                                phi_1844_ = true;
                            } else {
                                phi_1844_ = (_e1639_ == 0u);
                            }
                            let _e1643_ = phi_1844_;
                            if _e1138_ {
                                if _e1138_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1137_] = vec4<f32>(bitcast<f32>(select(_e1605_, 0u, _e1610_)), bitcast<f32>(select(_e1616_, 0u, _e1621_)), bitcast<f32>(select(_e1627_, 0u, _e1632_)), bitcast<f32>(select(_e1638_, 0u, _e1643_)));
                            }
                            let _e1650_ = local_8_;
                            let _e1652_ = bitcast<u32>(_e1650_.x);
                            let _e1653_ = (_e1652_ & 2139095040u);
                            if (_e1653_ == 2139095040u) {
                                phi_1873_ = true;
                            } else {
                                phi_1873_ = (_e1653_ == 0u);
                            }
                            let _e1657_ = phi_1873_;
                            let _e1661_ = local_9_;
                            let _e1663_ = bitcast<u32>(_e1661_.y);
                            let _e1664_ = (_e1663_ & 2139095040u);
                            if (_e1664_ == 2139095040u) {
                                phi_1890_ = true;
                            } else {
                                phi_1890_ = (_e1664_ == 0u);
                            }
                            let _e1668_ = phi_1890_;
                            let _e1672_ = local_10_;
                            let _e1674_ = bitcast<u32>(_e1672_.z);
                            let _e1675_ = (_e1674_ & 2139095040u);
                            if (_e1675_ == 2139095040u) {
                                phi_1907_ = true;
                            } else {
                                phi_1907_ = (_e1675_ == 0u);
                            }
                            let _e1679_ = phi_1907_;
                            let _e1683_ = local_11_;
                            let _e1685_ = bitcast<u32>(_e1683_.w);
                            let _e1686_ = (_e1685_ & 2139095040u);
                            if (_e1686_ == 2139095040u) {
                                phi_1924_ = true;
                            } else {
                                phi_1924_ = (_e1686_ == 0u);
                            }
                            let _e1690_ = phi_1924_;
                            if _e1145_ {
                                if _e1145_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1144_] = vec4<f32>(bitcast<f32>(select(_e1652_, 0u, _e1657_)), bitcast<f32>(select(_e1663_, 0u, _e1668_)), bitcast<f32>(select(_e1674_, 0u, _e1679_)), bitcast<f32>(select(_e1685_, 0u, _e1690_)));
                            }
                            let _e1697_ = local_12_;
                            let _e1699_ = bitcast<u32>(_e1697_.x);
                            let _e1700_ = (_e1699_ & 2139095040u);
                            if (_e1700_ == 2139095040u) {
                                phi_1953_ = true;
                            } else {
                                phi_1953_ = (_e1700_ == 0u);
                            }
                            let _e1704_ = phi_1953_;
                            let _e1708_ = local_13_;
                            let _e1710_ = bitcast<u32>(_e1708_.y);
                            let _e1711_ = (_e1710_ & 2139095040u);
                            if (_e1711_ == 2139095040u) {
                                phi_1970_ = true;
                            } else {
                                phi_1970_ = (_e1711_ == 0u);
                            }
                            let _e1715_ = phi_1970_;
                            let _e1719_ = local_14_;
                            let _e1721_ = bitcast<u32>(_e1719_.z);
                            let _e1722_ = (_e1721_ & 2139095040u);
                            if (_e1722_ == 2139095040u) {
                                phi_1987_ = true;
                            } else {
                                phi_1987_ = (_e1722_ == 0u);
                            }
                            let _e1726_ = phi_1987_;
                            let _e1730_ = local_15_;
                            let _e1732_ = bitcast<u32>(_e1730_.w);
                            let _e1733_ = (_e1732_ & 2139095040u);
                            if (_e1733_ == 2139095040u) {
                                phi_2004_ = true;
                            } else {
                                phi_2004_ = (_e1733_ == 0u);
                            }
                            let _e1737_ = phi_2004_;
                            if _e1152_ {
                                if _e1152_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1151_] = vec4<f32>(bitcast<f32>(select(_e1699_, 0u, _e1704_)), bitcast<f32>(select(_e1710_, 0u, _e1715_)), bitcast<f32>(select(_e1721_, 0u, _e1726_)), bitcast<f32>(select(_e1732_, 0u, _e1737_)));
                            }
                            let _e1744_ = local_16_;
                            let _e1746_ = bitcast<u32>(_e1744_.x);
                            let _e1747_ = (_e1746_ & 2139095040u);
                            if (_e1747_ == 2139095040u) {
                                phi_2033_ = true;
                            } else {
                                phi_2033_ = (_e1747_ == 0u);
                            }
                            let _e1751_ = phi_2033_;
                            let _e1755_ = local_17_;
                            let _e1757_ = bitcast<u32>(_e1755_.y);
                            let _e1758_ = (_e1757_ & 2139095040u);
                            if (_e1758_ == 2139095040u) {
                                phi_2050_ = true;
                            } else {
                                phi_2050_ = (_e1758_ == 0u);
                            }
                            let _e1762_ = phi_2050_;
                            let _e1766_ = local_18_;
                            let _e1768_ = bitcast<u32>(_e1766_.z);
                            let _e1769_ = (_e1768_ & 2139095040u);
                            if (_e1769_ == 2139095040u) {
                                phi_2067_ = true;
                            } else {
                                phi_2067_ = (_e1769_ == 0u);
                            }
                            let _e1773_ = phi_2067_;
                            let _e1777_ = local_19_;
                            let _e1779_ = bitcast<u32>(_e1777_.w);
                            let _e1780_ = (_e1779_ & 2139095040u);
                            if (_e1780_ == 2139095040u) {
                                phi_2084_ = true;
                            } else {
                                phi_2084_ = (_e1780_ == 0u);
                            }
                            let _e1784_ = phi_2084_;
                            if _e1159_ {
                                if _e1159_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1158_] = vec4<f32>(bitcast<f32>(select(_e1746_, 0u, _e1751_)), bitcast<f32>(select(_e1757_, 0u, _e1762_)), bitcast<f32>(select(_e1768_, 0u, _e1773_)), bitcast<f32>(select(_e1779_, 0u, _e1784_)));
                            }
                            let _e1791_ = local_20_;
                            let _e1793_ = bitcast<u32>(_e1791_.x);
                            let _e1794_ = (_e1793_ & 2139095040u);
                            if (_e1794_ == 2139095040u) {
                                phi_2113_ = true;
                            } else {
                                phi_2113_ = (_e1794_ == 0u);
                            }
                            let _e1798_ = phi_2113_;
                            let _e1802_ = local_21_;
                            let _e1804_ = bitcast<u32>(_e1802_.y);
                            let _e1805_ = (_e1804_ & 2139095040u);
                            if (_e1805_ == 2139095040u) {
                                phi_2130_ = true;
                            } else {
                                phi_2130_ = (_e1805_ == 0u);
                            }
                            let _e1809_ = phi_2130_;
                            let _e1813_ = local_22_;
                            let _e1815_ = bitcast<u32>(_e1813_.z);
                            let _e1816_ = (_e1815_ & 2139095040u);
                            if (_e1816_ == 2139095040u) {
                                phi_2147_ = true;
                            } else {
                                phi_2147_ = (_e1816_ == 0u);
                            }
                            let _e1820_ = phi_2147_;
                            let _e1824_ = local_23_;
                            let _e1826_ = bitcast<u32>(_e1824_.w);
                            let _e1827_ = (_e1826_ & 2139095040u);
                            if (_e1827_ == 2139095040u) {
                                phi_2164_ = true;
                            } else {
                                phi_2164_ = (_e1827_ == 0u);
                            }
                            let _e1831_ = phi_2164_;
                            if _e1166_ {
                                if _e1166_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1165_] = vec4<f32>(bitcast<f32>(select(_e1793_, 0u, _e1798_)), bitcast<f32>(select(_e1804_, 0u, _e1809_)), bitcast<f32>(select(_e1815_, 0u, _e1820_)), bitcast<f32>(select(_e1826_, 0u, _e1831_)));
                            }
                            let _e1838_ = local_24_;
                            let _e1840_ = bitcast<u32>(_e1838_.x);
                            let _e1841_ = (_e1840_ & 2139095040u);
                            if (_e1841_ == 2139095040u) {
                                phi_2193_ = true;
                            } else {
                                phi_2193_ = (_e1841_ == 0u);
                            }
                            let _e1845_ = phi_2193_;
                            let _e1849_ = local_25_;
                            let _e1851_ = bitcast<u32>(_e1849_.y);
                            let _e1852_ = (_e1851_ & 2139095040u);
                            if (_e1852_ == 2139095040u) {
                                phi_2210_ = true;
                            } else {
                                phi_2210_ = (_e1852_ == 0u);
                            }
                            let _e1856_ = phi_2210_;
                            let _e1860_ = local_26_;
                            let _e1862_ = bitcast<u32>(_e1860_.z);
                            let _e1863_ = (_e1862_ & 2139095040u);
                            if (_e1863_ == 2139095040u) {
                                phi_2227_ = true;
                            } else {
                                phi_2227_ = (_e1863_ == 0u);
                            }
                            let _e1867_ = phi_2227_;
                            let _e1871_ = local_27_;
                            let _e1873_ = bitcast<u32>(_e1871_.w);
                            let _e1874_ = (_e1873_ & 2139095040u);
                            if (_e1874_ == 2139095040u) {
                                phi_2244_ = true;
                            } else {
                                phi_2244_ = (_e1874_ == 0u);
                            }
                            let _e1878_ = phi_2244_;
                            if _e1173_ {
                                if _e1173_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                d_loss_d_features.member[_e1172_] = vec4<f32>(bitcast<f32>(select(_e1840_, 0u, _e1845_)), bitcast<f32>(select(_e1851_, 0u, _e1856_)), bitcast<f32>(select(_e1862_, 0u, _e1867_)), bitcast<f32>(select(_e1873_, 0u, _e1878_)));
                            }
                            phi_4367_ = _e1601_;
                        }
                        let _e1885_ = phi_4367_;
                        phi_4366_ = _e1885_;
                    }
                    let _e1887_ = phi_4366_;
                    phi_4362_ = _e1887_;
                    break;
                }
                case 2: {
                    let _e296_ = uniforms.member.member;
                    let _e299_ = uniforms.member.member_1_;
                    let _e301_ = select(1u, _e299_, (1u < _e299_));
                    let _e304_ = uniforms.member.member_2_;
                    let _e307_ = uniforms.member.member_6_;
                    let _e309_ = select(1u, _e307_, (1u < _e307_));
                    if (_e54_.x >= _e304_) {
                        phi_4364_ = false;
                    } else {
                        if (_e54_.y >= _e301_) {
                            phi_4365_ = false;
                        } else {
                            let _e312_ = (_e54_.x * 13u);
                            let _e313_ = (_e312_ < _e59_);
                            if _e313_ {
                                if _e313_ {
                                } else {
                                    phi_4362_ = true;
                                    break;
                                }
                                let _e316_ = tile_descriptors.member[_e312_];
                                phi_2302_ = _e316_;
                            } else {
                                phi_2302_ = vec4<f32>(0f, 0f, 0f, 0f);
                            }
                            let _e318_ = phi_2302_;
                            let _e320_ = bitcast<u32>(_e318_.x);
                            let _e322_ = bitcast<u32>(_e318_.y);
                            if (_e320_ < _e309_) {
                                let _e324_ = (_e322_ < _e309_);
                                let _e326_ = ((_e54_.y * 256u) + _e66_);
                                if _e324_ {
                                    let _e327_ = (_e326_ < _e296_);
                                    if _e327_ {
                                        let _e328_ = (_e54_.z * _e309_);
                                        let _e332_ = ((((_e328_ + _e320_) * _e296_) + _e326_) * 6u);
                                        let _e336_ = ((((_e328_ + _e322_) * _e296_) + _e326_) * 2u);
                                        let _e337_ = (_e332_ < _e61_);
                                        if _e337_ {
                                            if _e337_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e340_ = feature_outputs.member[_e332_];
                                            phi_2342_ = _e340_;
                                        } else {
                                            phi_2342_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e342_ = phi_2342_;
                                        let _e343_ = (_e332_ + 1u);
                                        let _e344_ = (_e343_ < _e61_);
                                        if _e344_ {
                                            if _e344_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e347_ = feature_outputs.member[_e343_];
                                            phi_2355_ = _e347_;
                                        } else {
                                            phi_2355_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e349_ = phi_2355_;
                                        let _e350_ = (_e332_ + 2u);
                                        let _e351_ = (_e350_ < _e61_);
                                        if _e351_ {
                                            if _e351_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e354_ = feature_outputs.member[_e350_];
                                            phi_2368_ = _e354_;
                                        } else {
                                            phi_2368_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e356_ = phi_2368_;
                                        let _e357_ = (_e332_ + 3u);
                                        let _e358_ = (_e357_ < _e61_);
                                        if _e358_ {
                                            if _e358_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e361_ = feature_outputs.member[_e357_];
                                            phi_2381_ = _e361_;
                                        } else {
                                            phi_2381_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e363_ = phi_2381_;
                                        let _e364_ = (_e332_ + 4u);
                                        let _e365_ = (_e364_ < _e61_);
                                        if _e365_ {
                                            if _e365_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e368_ = feature_outputs.member[_e364_];
                                            phi_2394_ = _e368_;
                                        } else {
                                            phi_2394_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e370_ = phi_2394_;
                                        let _e371_ = (_e332_ + 5u);
                                        let _e372_ = (_e371_ < _e61_);
                                        if _e372_ {
                                            if _e372_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e375_ = feature_outputs.member[_e371_];
                                            phi_2407_ = _e375_;
                                        } else {
                                            phi_2407_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e377_ = phi_2407_;
                                        let _e378_ = (_e336_ < _e57_);
                                        if _e378_ {
                                            if _e378_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e381_ = block_inputs_or_d_loss_d_inputs.member[_e336_];
                                            phi_2419_ = _e381_;
                                        } else {
                                            phi_2419_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e383_ = phi_2419_;
                                        let _e384_ = (_e336_ + 1u);
                                        let _e385_ = (_e384_ < _e57_);
                                        if _e385_ {
                                            if _e385_ {
                                            } else {
                                                phi_4362_ = true;
                                                break;
                                            }
                                            let _e388_ = block_inputs_or_d_loss_d_inputs.member[_e384_];
                                            phi_2432_ = _e388_;
                                        } else {
                                            phi_2432_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e390_ = phi_2432_;
                                        phi_2441_ = _e390_.w;
                                        phi_2442_ = _e390_.z;
                                        phi_2443_ = _e390_.y;
                                        phi_2444_ = _e390_.x;
                                        phi_2445_ = _e383_.w;
                                        phi_2446_ = _e383_.z;
                                        phi_2447_ = _e383_.y;
                                        phi_2448_ = _e383_.x;
                                        phi_2449_ = _e377_;
                                        phi_2450_ = _e370_;
                                        phi_2451_ = _e363_;
                                        phi_2452_ = _e356_;
                                        phi_2453_ = _e349_;
                                        phi_2454_ = _e342_;
                                    } else {
                                        phi_2441_ = f32();
                                        phi_2442_ = f32();
                                        phi_2443_ = f32();
                                        phi_2444_ = f32();
                                        phi_2445_ = f32();
                                        phi_2446_ = f32();
                                        phi_2447_ = f32();
                                        phi_2448_ = f32();
                                        phi_2449_ = vec4<f32>();
                                        phi_2450_ = vec4<f32>();
                                        phi_2451_ = vec4<f32>();
                                        phi_2452_ = vec4<f32>();
                                        phi_2453_ = vec4<f32>();
                                        phi_2454_ = vec4<f32>();
                                    }
                                    let _e400_ = phi_2441_;
                                    let _e402_ = phi_2442_;
                                    let _e404_ = phi_2443_;
                                    let _e406_ = phi_2444_;
                                    let _e408_ = phi_2445_;
                                    let _e410_ = phi_2446_;
                                    let _e412_ = phi_2447_;
                                    let _e414_ = phi_2448_;
                                    let _e416_ = phi_2449_;
                                    let _e418_ = phi_2450_;
                                    let _e420_ = phi_2451_;
                                    let _e422_ = phi_2452_;
                                    let _e424_ = phi_2453_;
                                    let _e426_ = phi_2454_;
                                    phi_2456_ = _e400_;
                                    phi_2457_ = _e402_;
                                    phi_2458_ = _e404_;
                                    phi_2459_ = _e406_;
                                    phi_2460_ = _e408_;
                                    phi_2461_ = _e410_;
                                    phi_2462_ = _e412_;
                                    phi_2463_ = _e414_;
                                    phi_2464_ = _e416_;
                                    phi_2465_ = _e418_;
                                    phi_2466_ = _e420_;
                                    phi_2467_ = _e422_;
                                    phi_2468_ = _e424_;
                                    phi_2469_ = _e426_;
                                    phi_2471_ = select(true, false, _e327_);
                                } else {
                                    phi_2456_ = f32();
                                    phi_2457_ = f32();
                                    phi_2458_ = f32();
                                    phi_2459_ = f32();
                                    phi_2460_ = f32();
                                    phi_2461_ = f32();
                                    phi_2462_ = f32();
                                    phi_2463_ = f32();
                                    phi_2464_ = vec4<f32>();
                                    phi_2465_ = vec4<f32>();
                                    phi_2466_ = vec4<f32>();
                                    phi_2467_ = vec4<f32>();
                                    phi_2468_ = vec4<f32>();
                                    phi_2469_ = vec4<f32>();
                                    phi_2471_ = false;
                                }
                                let _e429_ = phi_2456_;
                                let _e431_ = phi_2457_;
                                let _e433_ = phi_2458_;
                                let _e435_ = phi_2459_;
                                let _e437_ = phi_2460_;
                                let _e439_ = phi_2461_;
                                let _e441_ = phi_2462_;
                                let _e443_ = phi_2463_;
                                let _e445_ = phi_2464_;
                                let _e447_ = phi_2465_;
                                let _e449_ = phi_2466_;
                                let _e451_ = phi_2467_;
                                let _e453_ = phi_2468_;
                                let _e455_ = phi_2469_;
                                let _e457_ = phi_2471_;
                                phi_2472_ = _e429_;
                                phi_2473_ = _e431_;
                                phi_2474_ = _e433_;
                                phi_2475_ = _e435_;
                                phi_2476_ = _e437_;
                                phi_2477_ = _e439_;
                                phi_2478_ = _e441_;
                                phi_2479_ = _e443_;
                                phi_2480_ = _e445_;
                                phi_2481_ = _e447_;
                                phi_2482_ = _e449_;
                                phi_2483_ = _e451_;
                                phi_2484_ = _e453_;
                                phi_2485_ = _e455_;
                                phi_2486_ = select(true, false, _e324_);
                                phi_2487_ = _e457_;
                            } else {
                                phi_2472_ = f32();
                                phi_2473_ = f32();
                                phi_2474_ = f32();
                                phi_2475_ = f32();
                                phi_2476_ = f32();
                                phi_2477_ = f32();
                                phi_2478_ = f32();
                                phi_2479_ = f32();
                                phi_2480_ = vec4<f32>();
                                phi_2481_ = vec4<f32>();
                                phi_2482_ = vec4<f32>();
                                phi_2483_ = vec4<f32>();
                                phi_2484_ = vec4<f32>();
                                phi_2485_ = vec4<f32>();
                                phi_2486_ = true;
                                phi_2487_ = false;
                            }
                            let _e460_ = phi_2472_;
                            let _e462_ = phi_2473_;
                            let _e464_ = phi_2474_;
                            let _e466_ = phi_2475_;
                            let _e468_ = phi_2476_;
                            let _e470_ = phi_2477_;
                            let _e472_ = phi_2478_;
                            let _e474_ = phi_2479_;
                            let _e476_ = phi_2480_;
                            let _e478_ = phi_2481_;
                            let _e480_ = phi_2482_;
                            let _e482_ = phi_2483_;
                            let _e484_ = phi_2484_;
                            let _e486_ = phi_2485_;
                            let _e488_ = phi_2486_;
                            let _e490_ = phi_2487_;
                            let _e491_ = select(_e490_, true, _e488_);
                            let _e492_ = select(_e460_, 0f, _e491_);
                            let _e493_ = select(_e462_, 0f, _e491_);
                            let _e494_ = select(_e464_, 0f, _e491_);
                            let _e495_ = select(_e466_, 0f, _e491_);
                            let _e496_ = select(_e468_, 0f, _e491_);
                            let _e497_ = select(_e470_, 0f, _e491_);
                            let _e498_ = select(_e472_, 0f, _e491_);
                            let _e499_ = select(_e474_, 0f, _e491_);
                            let _e500_ = vec4(_e491_);
                            let _e501_ = select(_e476_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e502_ = select(_e478_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e503_ = select(_e480_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e504_ = select(_e482_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e505_ = select(_e484_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e506_ = select(_e486_, vec4<f32>(0f, 0f, 0f, 0f), _e500_);
                            let _e512_ = ((_e304_ + ((((_e54_.z * _e304_) + _e54_.x) * _e301_) + _e54_.y)) * 48u);
                            phi_4173_ = false;
                            phi_2525_ = type_19_(0u, 4u);
                            loop {
                                let _e514_ = phi_4173_;
                                let _e516_ = phi_2525_;
                                if (_e516_.member < _e516_.member_1_) {
                                    phi_2526_ = type_19_((_e516_.member + 1u), _e516_.member_1_);
                                    phi_2549_ = type_19_(1u, _e516_.member);
                                } else {
                                    phi_2526_ = _e516_;
                                    phi_2549_ = type_19_(0u, u32());
                                }
                                let _e529_ = phi_2526_;
                                let _e531_ = phi_2549_;
                                let _e535_ = (bitcast<i32>(_e531_.member) != 0i);
                                if _e535_ {
                                    let _e536_ = (_e531_.member_1_ & 3u);
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_2567_ = _e504_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2567_ = _e504_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2567_ = _e504_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2567_ = _e504_.w;
                                            break;
                                        }
                                    }
                                    let _e543_ = phi_2567_;
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_2578_ = _e503_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2578_ = _e503_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2578_ = _e503_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2578_ = _e503_.w;
                                            break;
                                        }
                                    }
                                    let _e550_ = phi_2578_;
                                    let _e557_ = bitcast<u32>(((_e543_ * _e499_) + (_e550_ * _e495_)));
                                    let _e558_ = (_e557_ & 2139095040u);
                                    if (_e558_ == 2139095040u) {
                                        phi_2595_ = true;
                                    } else {
                                        phi_2595_ = (_e558_ == 0u);
                                    }
                                    let _e562_ = phi_2595_;
                                    let _e565_ = bitcast<u32>(((_e543_ * _e498_) + (_e550_ * _e494_)));
                                    let _e566_ = (_e565_ & 2139095040u);
                                    if (_e566_ == 2139095040u) {
                                        phi_2611_ = true;
                                    } else {
                                        phi_2611_ = (_e566_ == 0u);
                                    }
                                    let _e570_ = phi_2611_;
                                    let _e574_ = select(255u, _e66_, (_e66_ < 255u));
                                    let _e575_ = (_e574_ < 256u);
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e557_, 0u, _e562_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e565_, 0u, _e570_));
                                    workgroupBarrier();
                                    phi_2641_ = 128u;
                                    loop {
                                        let _e579_ = phi_2641_;
                                        let _e580_ = (_e579_ > 0u);
                                        if _e580_ {
                                            if (_e66_ < _e579_) {
                                                let _e582_ = (_e66_ + _e579_);
                                                let _e583_ = (_e582_ < 256u);
                                                if _e583_ {
                                                } else {
                                                    phi_4161_ = true;
                                                    break;
                                                }
                                                let _e585_ = reduce_a[_e582_];
                                                let _e586_ = (_e66_ < 256u);
                                                if _e586_ {
                                                } else {
                                                    phi_4161_ = true;
                                                    break;
                                                }
                                                let _e588_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e588_ + _e585_);
                                                if _e583_ {
                                                } else {
                                                    phi_4161_ = true;
                                                    break;
                                                }
                                                let _e591_ = reduce_b[_e582_];
                                                if _e586_ {
                                                } else {
                                                    phi_4161_ = true;
                                                    break;
                                                }
                                                let _e593_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e593_ + _e591_);
                                            }
                                            workgroupBarrier();
                                            phi_2642_ = (_e579_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2642_ = u32();
                                        }
                                        let _e598_ = phi_2642_;
                                        phi_2641_ = _e598_;
                                        phi_4161_ = _e514_;
                                        if !(_e580_) {
                                            break;
                                        }
                                    }
                                    let _e601_ = phi_4161_;
                                    phi_4265_ = _e601_;
                                    if _e601_ {
                                        break;
                                    }
                                    let _e603_ = reduce_a[0];
                                    let _e604_ = bitcast<u32>(_e603_);
                                    let _e605_ = (_e604_ & 2139095040u);
                                    if (_e605_ == 2139095040u) {
                                        phi_2702_ = true;
                                    } else {
                                        phi_2702_ = (_e605_ == 0u);
                                    }
                                    let _e609_ = phi_2702_;
                                    let _e613_ = reduce_b[0];
                                    let _e614_ = bitcast<u32>(_e613_);
                                    let _e615_ = (_e614_ & 2139095040u);
                                    if (_e615_ == 2139095040u) {
                                        phi_2720_ = true;
                                    } else {
                                        phi_2720_ = (_e615_ == 0u);
                                    }
                                    let _e619_ = phi_2720_;
                                    workgroupBarrier();
                                    let _e628_ = bitcast<u32>(((_e543_ * _e497_) + (_e550_ * _e493_)));
                                    let _e629_ = (_e628_ & 2139095040u);
                                    if (_e629_ == 2139095040u) {
                                        phi_2742_ = true;
                                    } else {
                                        phi_2742_ = (_e629_ == 0u);
                                    }
                                    let _e633_ = phi_2742_;
                                    let _e636_ = bitcast<u32>(((_e543_ * _e496_) + (_e550_ * _e492_)));
                                    let _e637_ = (_e636_ & 2139095040u);
                                    if (_e637_ == 2139095040u) {
                                        phi_2758_ = true;
                                    } else {
                                        phi_2758_ = (_e637_ == 0u);
                                    }
                                    let _e641_ = phi_2758_;
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e628_, 0u, _e633_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e636_, 0u, _e641_));
                                    workgroupBarrier();
                                    phi_2788_ = 128u;
                                    loop {
                                        let _e645_ = phi_2788_;
                                        let _e646_ = (_e645_ > 0u);
                                        if _e646_ {
                                            if (_e66_ < _e645_) {
                                                let _e648_ = (_e66_ + _e645_);
                                                let _e649_ = (_e648_ < 256u);
                                                if _e649_ {
                                                } else {
                                                    phi_4191_ = true;
                                                    break;
                                                }
                                                let _e651_ = reduce_a[_e648_];
                                                let _e652_ = (_e66_ < 256u);
                                                if _e652_ {
                                                } else {
                                                    phi_4191_ = true;
                                                    break;
                                                }
                                                let _e654_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e654_ + _e651_);
                                                if _e649_ {
                                                } else {
                                                    phi_4191_ = true;
                                                    break;
                                                }
                                                let _e657_ = reduce_b[_e648_];
                                                if _e652_ {
                                                } else {
                                                    phi_4191_ = true;
                                                    break;
                                                }
                                                let _e659_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e659_ + _e657_);
                                            }
                                            workgroupBarrier();
                                            phi_2789_ = (_e645_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2789_ = u32();
                                        }
                                        let _e664_ = phi_2789_;
                                        phi_2788_ = _e664_;
                                        phi_4191_ = _e601_;
                                        if !(_e646_) {
                                            break;
                                        }
                                    }
                                    let _e667_ = phi_4191_;
                                    phi_4265_ = _e667_;
                                    if _e667_ {
                                        break;
                                    }
                                    let _e668_ = reduce_a[0];
                                    let _e669_ = bitcast<u32>(_e668_);
                                    let _e670_ = (_e669_ & 2139095040u);
                                    if (_e670_ == 2139095040u) {
                                        phi_2849_ = true;
                                    } else {
                                        phi_2849_ = (_e670_ == 0u);
                                    }
                                    let _e674_ = phi_2849_;
                                    let _e677_ = reduce_b[0];
                                    let _e678_ = bitcast<u32>(_e677_);
                                    let _e679_ = (_e678_ & 2139095040u);
                                    if (_e679_ == 2139095040u) {
                                        phi_2867_ = true;
                                    } else {
                                        phi_2867_ = (_e679_ == 0u);
                                    }
                                    let _e683_ = phi_2867_;
                                    workgroupBarrier();
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_2883_ = _e502_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2883_ = _e502_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2883_ = _e502_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2883_ = _e502_.w;
                                            break;
                                        }
                                    }
                                    let _e692_ = phi_2883_;
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_2894_ = _e501_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_2894_ = _e501_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_2894_ = _e501_.z;
                                            break;
                                        }
                                        default: {
                                            phi_2894_ = _e501_.w;
                                            break;
                                        }
                                    }
                                    let _e699_ = phi_2894_;
                                    let _e706_ = bitcast<u32>(((_e692_ * _e499_) + (_e699_ * _e495_)));
                                    let _e707_ = (_e706_ & 2139095040u);
                                    if (_e707_ == 2139095040u) {
                                        phi_2911_ = true;
                                    } else {
                                        phi_2911_ = (_e707_ == 0u);
                                    }
                                    let _e711_ = phi_2911_;
                                    let _e714_ = bitcast<u32>(((_e692_ * _e498_) + (_e699_ * _e494_)));
                                    let _e715_ = (_e714_ & 2139095040u);
                                    if (_e715_ == 2139095040u) {
                                        phi_2927_ = true;
                                    } else {
                                        phi_2927_ = (_e715_ == 0u);
                                    }
                                    let _e719_ = phi_2927_;
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e706_, 0u, _e711_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e714_, 0u, _e719_));
                                    workgroupBarrier();
                                    phi_2957_ = 128u;
                                    loop {
                                        let _e723_ = phi_2957_;
                                        let _e724_ = (_e723_ > 0u);
                                        if _e724_ {
                                            if (_e66_ < _e723_) {
                                                let _e726_ = (_e66_ + _e723_);
                                                let _e727_ = (_e726_ < 256u);
                                                if _e727_ {
                                                } else {
                                                    phi_4205_ = true;
                                                    break;
                                                }
                                                let _e729_ = reduce_a[_e726_];
                                                let _e730_ = (_e66_ < 256u);
                                                if _e730_ {
                                                } else {
                                                    phi_4205_ = true;
                                                    break;
                                                }
                                                let _e732_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e732_ + _e729_);
                                                if _e727_ {
                                                } else {
                                                    phi_4205_ = true;
                                                    break;
                                                }
                                                let _e735_ = reduce_b[_e726_];
                                                if _e730_ {
                                                } else {
                                                    phi_4205_ = true;
                                                    break;
                                                }
                                                let _e737_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e737_ + _e735_);
                                            }
                                            workgroupBarrier();
                                            phi_2958_ = (_e723_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_2958_ = u32();
                                        }
                                        let _e742_ = phi_2958_;
                                        phi_2957_ = _e742_;
                                        phi_4205_ = _e667_;
                                        if !(_e724_) {
                                            break;
                                        }
                                    }
                                    let _e745_ = phi_4205_;
                                    phi_4265_ = _e745_;
                                    if _e745_ {
                                        break;
                                    }
                                    let _e746_ = reduce_a[0];
                                    let _e747_ = bitcast<u32>(_e746_);
                                    let _e748_ = (_e747_ & 2139095040u);
                                    if (_e748_ == 2139095040u) {
                                        phi_3018_ = true;
                                    } else {
                                        phi_3018_ = (_e748_ == 0u);
                                    }
                                    let _e752_ = phi_3018_;
                                    let _e755_ = reduce_b[0];
                                    let _e756_ = bitcast<u32>(_e755_);
                                    let _e757_ = (_e756_ & 2139095040u);
                                    if (_e757_ == 2139095040u) {
                                        phi_3036_ = true;
                                    } else {
                                        phi_3036_ = (_e757_ == 0u);
                                    }
                                    let _e761_ = phi_3036_;
                                    workgroupBarrier();
                                    let _e770_ = bitcast<u32>(((_e692_ * _e497_) + (_e699_ * _e493_)));
                                    let _e771_ = (_e770_ & 2139095040u);
                                    if (_e771_ == 2139095040u) {
                                        phi_3058_ = true;
                                    } else {
                                        phi_3058_ = (_e771_ == 0u);
                                    }
                                    let _e775_ = phi_3058_;
                                    let _e778_ = bitcast<u32>(((_e692_ * _e496_) + (_e699_ * _e492_)));
                                    let _e779_ = (_e778_ & 2139095040u);
                                    if (_e779_ == 2139095040u) {
                                        phi_3074_ = true;
                                    } else {
                                        phi_3074_ = (_e779_ == 0u);
                                    }
                                    let _e783_ = phi_3074_;
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e770_, 0u, _e775_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e778_, 0u, _e783_));
                                    workgroupBarrier();
                                    phi_3104_ = 128u;
                                    loop {
                                        let _e787_ = phi_3104_;
                                        let _e788_ = (_e787_ > 0u);
                                        if _e788_ {
                                            if (_e66_ < _e787_) {
                                                let _e790_ = (_e66_ + _e787_);
                                                let _e791_ = (_e790_ < 256u);
                                                if _e791_ {
                                                } else {
                                                    phi_4221_ = true;
                                                    break;
                                                }
                                                let _e793_ = reduce_a[_e790_];
                                                let _e794_ = (_e66_ < 256u);
                                                if _e794_ {
                                                } else {
                                                    phi_4221_ = true;
                                                    break;
                                                }
                                                let _e796_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e796_ + _e793_);
                                                if _e791_ {
                                                } else {
                                                    phi_4221_ = true;
                                                    break;
                                                }
                                                let _e799_ = reduce_b[_e790_];
                                                if _e794_ {
                                                } else {
                                                    phi_4221_ = true;
                                                    break;
                                                }
                                                let _e801_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e801_ + _e799_);
                                            }
                                            workgroupBarrier();
                                            phi_3105_ = (_e787_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3105_ = u32();
                                        }
                                        let _e806_ = phi_3105_;
                                        phi_3104_ = _e806_;
                                        phi_4221_ = _e745_;
                                        if !(_e788_) {
                                            break;
                                        }
                                    }
                                    let _e809_ = phi_4221_;
                                    phi_4265_ = _e809_;
                                    if _e809_ {
                                        break;
                                    }
                                    let _e810_ = reduce_a[0];
                                    let _e811_ = bitcast<u32>(_e810_);
                                    let _e812_ = (_e811_ & 2139095040u);
                                    if (_e812_ == 2139095040u) {
                                        phi_3165_ = true;
                                    } else {
                                        phi_3165_ = (_e812_ == 0u);
                                    }
                                    let _e816_ = phi_3165_;
                                    let _e819_ = reduce_b[0];
                                    let _e820_ = bitcast<u32>(_e819_);
                                    let _e821_ = (_e820_ & 2139095040u);
                                    if (_e821_ == 2139095040u) {
                                        phi_3183_ = true;
                                    } else {
                                        phi_3183_ = (_e821_ == 0u);
                                    }
                                    let _e825_ = phi_3183_;
                                    workgroupBarrier();
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_3199_ = _e506_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_3199_ = _e506_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_3199_ = _e506_.z;
                                            break;
                                        }
                                        default: {
                                            phi_3199_ = _e506_.w;
                                            break;
                                        }
                                    }
                                    let _e834_ = phi_3199_;
                                    switch bitcast<i32>(_e536_) {
                                        case 0: {
                                            phi_3210_ = _e505_.x;
                                            break;
                                        }
                                        case 1: {
                                            phi_3210_ = _e505_.y;
                                            break;
                                        }
                                        case 2: {
                                            phi_3210_ = _e505_.z;
                                            break;
                                        }
                                        default: {
                                            phi_3210_ = _e505_.w;
                                            break;
                                        }
                                    }
                                    let _e841_ = phi_3210_;
                                    let _e848_ = bitcast<u32>(((_e834_ * _e499_) + (_e841_ * _e495_)));
                                    let _e849_ = (_e848_ & 2139095040u);
                                    if (_e849_ == 2139095040u) {
                                        phi_3227_ = true;
                                    } else {
                                        phi_3227_ = (_e849_ == 0u);
                                    }
                                    let _e853_ = phi_3227_;
                                    let _e856_ = bitcast<u32>(((_e834_ * _e498_) + (_e841_ * _e494_)));
                                    let _e857_ = (_e856_ & 2139095040u);
                                    if (_e857_ == 2139095040u) {
                                        phi_3243_ = true;
                                    } else {
                                        phi_3243_ = (_e857_ == 0u);
                                    }
                                    let _e861_ = phi_3243_;
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e848_, 0u, _e853_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e856_, 0u, _e861_));
                                    workgroupBarrier();
                                    phi_3273_ = 128u;
                                    loop {
                                        let _e865_ = phi_3273_;
                                        let _e866_ = (_e865_ > 0u);
                                        if _e866_ {
                                            if (_e66_ < _e865_) {
                                                let _e868_ = (_e66_ + _e865_);
                                                let _e869_ = (_e868_ < 256u);
                                                if _e869_ {
                                                } else {
                                                    phi_4235_ = true;
                                                    break;
                                                }
                                                let _e871_ = reduce_a[_e868_];
                                                let _e872_ = (_e66_ < 256u);
                                                if _e872_ {
                                                } else {
                                                    phi_4235_ = true;
                                                    break;
                                                }
                                                let _e874_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e874_ + _e871_);
                                                if _e869_ {
                                                } else {
                                                    phi_4235_ = true;
                                                    break;
                                                }
                                                let _e877_ = reduce_b[_e868_];
                                                if _e872_ {
                                                } else {
                                                    phi_4235_ = true;
                                                    break;
                                                }
                                                let _e879_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e879_ + _e877_);
                                            }
                                            workgroupBarrier();
                                            phi_3274_ = (_e865_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3274_ = u32();
                                        }
                                        let _e884_ = phi_3274_;
                                        phi_3273_ = _e884_;
                                        phi_4235_ = _e809_;
                                        if !(_e866_) {
                                            break;
                                        }
                                    }
                                    let _e887_ = phi_4235_;
                                    phi_4265_ = _e887_;
                                    if _e887_ {
                                        break;
                                    }
                                    let _e888_ = reduce_a[0];
                                    let _e889_ = bitcast<u32>(_e888_);
                                    let _e890_ = (_e889_ & 2139095040u);
                                    if (_e890_ == 2139095040u) {
                                        phi_3334_ = true;
                                    } else {
                                        phi_3334_ = (_e890_ == 0u);
                                    }
                                    let _e894_ = phi_3334_;
                                    let _e897_ = reduce_b[0];
                                    let _e898_ = bitcast<u32>(_e897_);
                                    let _e899_ = (_e898_ & 2139095040u);
                                    if (_e899_ == 2139095040u) {
                                        phi_3352_ = true;
                                    } else {
                                        phi_3352_ = (_e899_ == 0u);
                                    }
                                    let _e903_ = phi_3352_;
                                    workgroupBarrier();
                                    let _e912_ = bitcast<u32>(((_e834_ * _e497_) + (_e841_ * _e493_)));
                                    let _e913_ = (_e912_ & 2139095040u);
                                    if (_e913_ == 2139095040u) {
                                        phi_3374_ = true;
                                    } else {
                                        phi_3374_ = (_e913_ == 0u);
                                    }
                                    let _e917_ = phi_3374_;
                                    let _e920_ = bitcast<u32>(((_e834_ * _e496_) + (_e841_ * _e492_)));
                                    let _e921_ = (_e920_ & 2139095040u);
                                    if (_e921_ == 2139095040u) {
                                        phi_3390_ = true;
                                    } else {
                                        phi_3390_ = (_e921_ == 0u);
                                    }
                                    let _e925_ = phi_3390_;
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_a[_e574_] = bitcast<f32>(select(_e912_, 0u, _e917_));
                                    if _e575_ {
                                    } else {
                                        phi_4265_ = true;
                                        break;
                                    }
                                    reduce_b[_e574_] = bitcast<f32>(select(_e920_, 0u, _e925_));
                                    workgroupBarrier();
                                    phi_3420_ = 128u;
                                    loop {
                                        let _e929_ = phi_3420_;
                                        let _e930_ = (_e929_ > 0u);
                                        if _e930_ {
                                            if (_e66_ < _e929_) {
                                                let _e932_ = (_e66_ + _e929_);
                                                let _e933_ = (_e932_ < 256u);
                                                if _e933_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e935_ = reduce_a[_e932_];
                                                let _e936_ = (_e66_ < 256u);
                                                if _e936_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e938_ = reduce_a[_e66_];
                                                reduce_a[_e66_] = (_e938_ + _e935_);
                                                if _e933_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e941_ = reduce_b[_e932_];
                                                if _e936_ {
                                                } else {
                                                    phi_4251_ = true;
                                                    break;
                                                }
                                                let _e943_ = reduce_b[_e66_];
                                                reduce_b[_e66_] = (_e943_ + _e941_);
                                            }
                                            workgroupBarrier();
                                            phi_3421_ = (_e929_ >> bitcast<u32>(1i));
                                        } else {
                                            phi_3421_ = u32();
                                        }
                                        let _e948_ = phi_3421_;
                                        phi_3420_ = _e948_;
                                        phi_4251_ = _e887_;
                                        if !(_e930_) {
                                            break;
                                        }
                                    }
                                    let _e951_ = phi_4251_;
                                    phi_4265_ = _e951_;
                                    if _e951_ {
                                        break;
                                    }
                                    let _e952_ = reduce_a[0];
                                    let _e953_ = bitcast<u32>(_e952_);
                                    let _e954_ = (_e953_ & 2139095040u);
                                    if (_e954_ == 2139095040u) {
                                        phi_3481_ = true;
                                    } else {
                                        phi_3481_ = (_e954_ == 0u);
                                    }
                                    let _e958_ = phi_3481_;
                                    let _e961_ = reduce_b[0];
                                    let _e962_ = bitcast<u32>(_e961_);
                                    let _e963_ = (_e962_ & 2139095040u);
                                    if (_e963_ == 2139095040u) {
                                        phi_3499_ = true;
                                    } else {
                                        phi_3499_ = (_e963_ == 0u);
                                    }
                                    let _e967_ = phi_3499_;
                                    workgroupBarrier();
                                    if (_e66_ == 0u) {
                                        let _e971_ = (_e531_.member_1_ * 4u);
                                        let _e972_ = (_e512_ + _e971_);
                                        let _e973_ = (_e63_ == 0u);
                                        if _e973_ {
                                            phi_3521_ = 0u;
                                        } else {
                                            let _e974_ = (_e63_ - 1u);
                                            phi_3521_ = select(_e974_, _e972_, (_e972_ < _e974_));
                                        }
                                        let _e978_ = phi_3521_;
                                        if (_e978_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e978_] = bitcast<f32>(select(_e604_, 0u, _e609_));
                                        let _e982_ = (_e972_ + 1u);
                                        if _e973_ {
                                            phi_3540_ = 0u;
                                        } else {
                                            let _e983_ = (_e63_ - 1u);
                                            phi_3540_ = select(_e983_, _e982_, (_e982_ < _e983_));
                                        }
                                        let _e987_ = phi_3540_;
                                        if (_e987_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e987_] = bitcast<f32>(select(_e614_, 0u, _e619_));
                                        let _e991_ = (_e972_ + 2u);
                                        if _e973_ {
                                            phi_3559_ = 0u;
                                        } else {
                                            let _e992_ = (_e63_ - 1u);
                                            phi_3559_ = select(_e992_, _e991_, (_e991_ < _e992_));
                                        }
                                        let _e996_ = phi_3559_;
                                        if (_e996_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e996_] = bitcast<f32>(select(_e669_, 0u, _e674_));
                                        let _e1000_ = (_e972_ + 3u);
                                        if _e973_ {
                                            phi_3578_ = 0u;
                                        } else {
                                            let _e1001_ = (_e63_ - 1u);
                                            phi_3578_ = select(_e1001_, _e1000_, (_e1000_ < _e1001_));
                                        }
                                        let _e1005_ = phi_3578_;
                                        if (_e1005_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1005_] = bitcast<f32>(select(_e678_, 0u, _e683_));
                                        let _e1010_ = ((_e512_ + 16u) + _e971_);
                                        if _e973_ {
                                            phi_3597_ = 0u;
                                        } else {
                                            let _e1011_ = (_e63_ - 1u);
                                            phi_3597_ = select(_e1011_, _e1010_, (_e1010_ < _e1011_));
                                        }
                                        let _e1015_ = phi_3597_;
                                        if (_e1015_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1015_] = bitcast<f32>(select(_e747_, 0u, _e752_));
                                        let _e1019_ = (_e1010_ + 1u);
                                        if _e973_ {
                                            phi_3617_ = 0u;
                                        } else {
                                            let _e1020_ = (_e63_ - 1u);
                                            phi_3617_ = select(_e1020_, _e1019_, (_e1019_ < _e1020_));
                                        }
                                        let _e1024_ = phi_3617_;
                                        if (_e1024_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1024_] = bitcast<f32>(select(_e756_, 0u, _e761_));
                                        let _e1028_ = (_e1010_ + 2u);
                                        if _e973_ {
                                            phi_3637_ = 0u;
                                        } else {
                                            let _e1029_ = (_e63_ - 1u);
                                            phi_3637_ = select(_e1029_, _e1028_, (_e1028_ < _e1029_));
                                        }
                                        let _e1033_ = phi_3637_;
                                        if (_e1033_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1033_] = bitcast<f32>(select(_e811_, 0u, _e816_));
                                        let _e1037_ = (_e1010_ + 3u);
                                        if _e973_ {
                                            phi_3657_ = 0u;
                                        } else {
                                            let _e1038_ = (_e63_ - 1u);
                                            phi_3657_ = select(_e1038_, _e1037_, (_e1037_ < _e1038_));
                                        }
                                        let _e1042_ = phi_3657_;
                                        if (_e1042_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1042_] = bitcast<f32>(select(_e820_, 0u, _e825_));
                                        let _e1047_ = ((_e512_ + 32u) + _e971_);
                                        if _e973_ {
                                            phi_3676_ = 0u;
                                        } else {
                                            let _e1048_ = (_e63_ - 1u);
                                            phi_3676_ = select(_e1048_, _e1047_, (_e1047_ < _e1048_));
                                        }
                                        let _e1052_ = phi_3676_;
                                        if (_e1052_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1052_] = bitcast<f32>(select(_e889_, 0u, _e894_));
                                        let _e1056_ = (_e1047_ + 1u);
                                        if _e973_ {
                                            phi_3696_ = 0u;
                                        } else {
                                            let _e1057_ = (_e63_ - 1u);
                                            phi_3696_ = select(_e1057_, _e1056_, (_e1056_ < _e1057_));
                                        }
                                        let _e1061_ = phi_3696_;
                                        if (_e1061_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1061_] = bitcast<f32>(select(_e898_, 0u, _e903_));
                                        let _e1065_ = (_e1047_ + 2u);
                                        if _e973_ {
                                            phi_3716_ = 0u;
                                        } else {
                                            let _e1066_ = (_e63_ - 1u);
                                            phi_3716_ = select(_e1066_, _e1065_, (_e1065_ < _e1066_));
                                        }
                                        let _e1070_ = phi_3716_;
                                        if (_e1070_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1070_] = bitcast<f32>(select(_e953_, 0u, _e958_));
                                        let _e1074_ = (_e1047_ + 3u);
                                        if _e973_ {
                                            phi_3736_ = 0u;
                                        } else {
                                            let _e1075_ = (_e63_ - 1u);
                                            phi_3736_ = select(_e1075_, _e1074_, (_e1074_ < _e1075_));
                                        }
                                        let _e1079_ = phi_3736_;
                                        if (_e1079_ < _e63_) {
                                        } else {
                                            phi_4265_ = true;
                                            break;
                                        }
                                        tile_weight_grads.member[_e1079_] = bitcast<f32>(select(_e962_, 0u, _e967_));
                                    }
                                    phi_4266_ = _e951_;
                                } else {
                                    phi_4266_ = _e514_;
                                }
                                let _e1084_ = phi_4266_;
                                phi_4173_ = _e1084_;
                                phi_2525_ = _e529_;
                                phi_4265_ = _e1084_;
                                if !(_e535_) {
                                    break;
                                }
                            }
                            let _e1087_ = phi_4265_;
                            phi_4362_ = _e1087_;
                            if _e1087_ {
                                break;
                            }
                            phi_4365_ = _e1087_;
                        }
                        let _e1089_ = phi_4365_;
                        phi_4364_ = _e1089_;
                    }
                    let _e1091_ = phi_4364_;
                    phi_4362_ = _e1091_;
                    break;
                }
                case 3: {
                    let _e74_ = uniforms.member.member_2_;
                    let _e77_ = uniforms.member.member_1_;
                    let _e79_ = select(1u, _e77_, (1u < _e77_));
                    let _e82_ = uniforms.member.member_4_;
                    if (_e54_.x >= _e74_) {
                        phi_4363_ = false;
                    } else {
                        let _e86_ = (_e54_.x * 48u);
                        phi_4127_ = false;
                        phi_3767_ = type_19_(0u, 24u);
                        loop {
                            let _e88_ = phi_4127_;
                            let _e90_ = phi_3767_;
                            if (_e90_.member < _e90_.member_1_) {
                                phi_3768_ = type_19_((_e90_.member + 1u), _e90_.member_1_);
                                phi_3791_ = type_19_(1u, _e90_.member);
                            } else {
                                phi_3768_ = _e90_;
                                phi_3791_ = type_19_(0u, u32());
                            }
                            let _e103_ = phi_3768_;
                            let _e105_ = phi_3791_;
                            let _e109_ = (bitcast<i32>(_e105_.member) != 0i);
                            if _e109_ {
                                let _e110_ = (_e105_.member_1_ * 2u);
                                let _e111_ = (_e110_ + 1u);
                                phi_4125_ = _e88_;
                                phi_3802_ = 0u;
                                phi_3805_ = 0f;
                                phi_3807_ = 0f;
                                loop {
                                    let _e113_ = phi_4125_;
                                    let _e115_ = phi_3802_;
                                    let _e117_ = phi_3805_;
                                    let _e119_ = phi_3807_;
                                    local_2_ = _e119_;
                                    local_3_ = _e117_;
                                    let _e120_ = (_e115_ < select(1u, _e82_, (1u < _e82_)));
                                    if _e120_ {
                                        phi_3816_ = _e66_;
                                        phi_3819_ = _e117_;
                                        phi_3821_ = _e119_;
                                        loop {
                                            let _e122_ = phi_3816_;
                                            let _e124_ = phi_3819_;
                                            let _e126_ = phi_3821_;
                                            local_36_ = _e124_;
                                            local_37_ = _e126_;
                                            let _e127_ = (_e122_ < _e79_);
                                            if _e127_ {
                                                let _e133_ = ((_e74_ + ((((_e115_ * _e74_) + _e54_.x) * _e79_) + _e122_)) * 48u);
                                                let _e134_ = (_e133_ + _e110_);
                                                let _e135_ = (_e63_ == 0u);
                                                if _e135_ {
                                                    phi_3846_ = 0u;
                                                } else {
                                                    let _e136_ = (_e63_ - 1u);
                                                    phi_3846_ = select(_e136_, _e134_, (_e134_ < _e136_));
                                                }
                                                let _e140_ = phi_3846_;
                                                if (_e140_ < _e63_) {
                                                } else {
                                                    phi_4122_ = true;
                                                    break;
                                                }
                                                let _e144_ = tile_weight_grads.member[_e140_];
                                                let _e146_ = (_e133_ + _e111_);
                                                if _e135_ {
                                                    phi_3866_ = 0u;
                                                } else {
                                                    let _e147_ = (_e63_ - 1u);
                                                    phi_3866_ = select(_e147_, _e146_, (_e146_ < _e147_));
                                                }
                                                let _e151_ = phi_3866_;
                                                if (_e151_ < _e63_) {
                                                } else {
                                                    phi_4122_ = true;
                                                    break;
                                                }
                                                let _e155_ = tile_weight_grads.member[_e151_];
                                                phi_3817_ = (_e122_ + 256u);
                                                phi_3820_ = (_e124_ + _e155_);
                                                phi_3822_ = (_e126_ + _e144_);
                                            } else {
                                                phi_3817_ = u32();
                                                phi_3820_ = f32();
                                                phi_3822_ = f32();
                                            }
                                            let _e159_ = phi_3817_;
                                            let _e161_ = phi_3820_;
                                            let _e163_ = phi_3822_;
                                            phi_3816_ = _e159_;
                                            phi_3819_ = _e161_;
                                            phi_3821_ = _e163_;
                                            phi_4122_ = _e113_;
                                            if !(_e127_) {
                                                break;
                                            }
                                        }
                                        let _e166_ = phi_4122_;
                                        phi_4134_ = _e166_;
                                        if _e166_ {
                                            break;
                                        }
                                        phi_4135_ = _e166_;
                                        phi_3803_ = (_e115_ + 1u);
                                        let _e2520_ = local_36_;
                                        phi_3806_ = _e2520_;
                                        let _e2523_ = local_37_;
                                        phi_3808_ = _e2523_;
                                    } else {
                                        phi_4135_ = _e113_;
                                        phi_3803_ = u32();
                                        phi_3806_ = f32();
                                        phi_3808_ = f32();
                                    }
                                    let _e169_ = phi_4135_;
                                    let _e171_ = phi_3803_;
                                    let _e173_ = phi_3806_;
                                    let _e175_ = phi_3808_;
                                    phi_4125_ = _e169_;
                                    phi_3802_ = _e171_;
                                    phi_3805_ = _e173_;
                                    phi_3807_ = _e175_;
                                    phi_4134_ = _e169_;
                                    if !(_e120_) {
                                        break;
                                    }
                                }
                                let _e178_ = phi_4134_;
                                phi_4146_ = _e178_;
                                if _e178_ {
                                    break;
                                }
                                let _e180_ = local_2_;
                                let _e181_ = bitcast<u32>(_e180_);
                                let _e182_ = (_e181_ & 2139095040u);
                                if (_e182_ == 2139095040u) {
                                    phi_3887_ = true;
                                } else {
                                    phi_3887_ = (_e182_ == 0u);
                                }
                                let _e186_ = phi_3887_;
                                let _e190_ = local_3_;
                                let _e191_ = bitcast<u32>(_e190_);
                                let _e192_ = (_e191_ & 2139095040u);
                                if (_e192_ == 2139095040u) {
                                    phi_3903_ = true;
                                } else {
                                    phi_3903_ = (_e192_ == 0u);
                                }
                                let _e196_ = phi_3903_;
                                let _e200_ = select(255u, _e66_, (_e66_ < 255u));
                                let _e201_ = (_e200_ < 256u);
                                if _e201_ {
                                } else {
                                    phi_4146_ = true;
                                    break;
                                }
                                reduce_a[_e200_] = bitcast<f32>(select(_e181_, 0u, _e186_));
                                if _e201_ {
                                } else {
                                    phi_4146_ = true;
                                    break;
                                }
                                reduce_b[_e200_] = bitcast<f32>(select(_e191_, 0u, _e196_));
                                workgroupBarrier();
                                phi_3933_ = 128u;
                                loop {
                                    let _e205_ = phi_3933_;
                                    let _e206_ = (_e205_ > 0u);
                                    if _e206_ {
                                        if (_e66_ < _e205_) {
                                            let _e208_ = (_e66_ + _e205_);
                                            let _e209_ = (_e208_ < 256u);
                                            if _e209_ {
                                            } else {
                                                phi_4136_ = true;
                                                break;
                                            }
                                            let _e211_ = reduce_a[_e208_];
                                            let _e212_ = (_e66_ < 256u);
                                            if _e212_ {
                                            } else {
                                                phi_4136_ = true;
                                                break;
                                            }
                                            let _e214_ = reduce_a[_e66_];
                                            reduce_a[_e66_] = (_e214_ + _e211_);
                                            if _e209_ {
                                            } else {
                                                phi_4136_ = true;
                                                break;
                                            }
                                            let _e217_ = reduce_b[_e208_];
                                            if _e212_ {
                                            } else {
                                                phi_4136_ = true;
                                                break;
                                            }
                                            let _e219_ = reduce_b[_e66_];
                                            reduce_b[_e66_] = (_e219_ + _e217_);
                                        }
                                        workgroupBarrier();
                                        phi_3934_ = (_e205_ >> bitcast<u32>(1i));
                                    } else {
                                        phi_3934_ = u32();
                                    }
                                    let _e224_ = phi_3934_;
                                    phi_3933_ = _e224_;
                                    phi_4136_ = _e178_;
                                    if !(_e206_) {
                                        break;
                                    }
                                }
                                let _e227_ = phi_4136_;
                                phi_4146_ = _e227_;
                                if _e227_ {
                                    break;
                                }
                                let _e229_ = reduce_a[0];
                                let _e230_ = bitcast<u32>(_e229_);
                                let _e231_ = (_e230_ & 2139095040u);
                                if (_e231_ == 2139095040u) {
                                    phi_3994_ = true;
                                } else {
                                    phi_3994_ = (_e231_ == 0u);
                                }
                                let _e235_ = phi_3994_;
                                let _e239_ = reduce_b[0];
                                let _e240_ = bitcast<u32>(_e239_);
                                let _e241_ = (_e240_ & 2139095040u);
                                if (_e241_ == 2139095040u) {
                                    phi_4012_ = true;
                                } else {
                                    phi_4012_ = (_e241_ == 0u);
                                }
                                let _e245_ = phi_4012_;
                                workgroupBarrier();
                                if (_e66_ == 0u) {
                                    let _e249_ = bitcast<u32>(bitcast<f32>(select(_e230_, 0u, _e235_)));
                                    let _e250_ = (_e249_ & 2139095040u);
                                    if (_e250_ == 2139095040u) {
                                        phi_4032_ = true;
                                    } else {
                                        phi_4032_ = (_e250_ == 0u);
                                    }
                                    let _e254_ = phi_4032_;
                                    let _e257_ = (_e86_ + _e110_);
                                    let _e258_ = (_e63_ == 0u);
                                    if _e258_ {
                                        phi_4049_ = 0u;
                                    } else {
                                        let _e259_ = (_e63_ - 1u);
                                        phi_4049_ = select(_e259_, _e257_, (_e257_ < _e259_));
                                    }
                                    let _e263_ = phi_4049_;
                                    if (_e263_ < _e63_) {
                                    } else {
                                        phi_4146_ = true;
                                        break;
                                    }
                                    tile_weight_grads.member[_e263_] = bitcast<f32>(select(_e249_, 0u, _e254_));
                                    let _e267_ = bitcast<u32>(bitcast<f32>(select(_e240_, 0u, _e245_)));
                                    let _e268_ = (_e267_ & 2139095040u);
                                    if (_e268_ == 2139095040u) {
                                        phi_4066_ = true;
                                    } else {
                                        phi_4066_ = (_e268_ == 0u);
                                    }
                                    let _e272_ = phi_4066_;
                                    let _e275_ = (_e86_ + _e111_);
                                    if _e258_ {
                                        phi_4083_ = 0u;
                                    } else {
                                        let _e276_ = (_e63_ - 1u);
                                        phi_4083_ = select(_e276_, _e275_, (_e275_ < _e276_));
                                    }
                                    let _e280_ = phi_4083_;
                                    if (_e280_ < _e63_) {
                                    } else {
                                        phi_4146_ = true;
                                        break;
                                    }
                                    tile_weight_grads.member[_e280_] = bitcast<f32>(select(_e267_, 0u, _e272_));
                                }
                                phi_4147_ = _e227_;
                            } else {
                                phi_4147_ = _e88_;
                            }
                            let _e285_ = phi_4147_;
                            phi_4127_ = _e285_;
                            phi_3767_ = _e103_;
                            phi_4146_ = _e285_;
                            if !(_e109_) {
                                break;
                            }
                        }
                        let _e288_ = phi_4146_;
                        phi_4362_ = _e288_;
                        if _e288_ {
                            break;
                        }
                        phi_4363_ = _e288_;
                    }
                    let _e290_ = phi_4363_;
                    phi_4362_ = _e290_;
                    break;
                }
                default: {
                    phi_4362_ = false;
                    break;
                }
            }
            let _e2498_ = phi_4362_;
            if _e2498_ {
                break;
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn tile_mix_fwd_adj(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}
