struct type_5_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: f32,
    member_7_: u32,
}

struct type_6_ {
    member: type_5_,
}

struct type_9_ {
    member: array<f32>,
}

struct type_13_ {
    member: array<vec2<f32>>,
}

struct type_17_ {
    member: array<vec4<f32>>,
}

struct type_30_ {
    member: u32,
    member_1_: u32,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> sg_lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_6_;
@group(0) @binding(1) 
var<storage, read_write> theta: type_9_;
@group(0) @binding(2) 
var<storage> grad: type_9_;
@group(0) @binding(3) 
var<storage> param_bounds: type_13_;
@group(0) @binding(4) 
var<storage, read_write> lbfgs_s_y: type_17_;
@group(0) @binding(5) 
var<storage, read_write> search_dir: type_9_;
var<workgroup> dot_scratch: array<f32, 256>;
var<workgroup> scratch_q: array<f32, 256>;
var<workgroup> lbfgs_scalars: array<f32, 32>;

fn function_() {
    var phi_3006_: f32;
    var phi_3007_: bool;
    var phi_3024_: u32;
    var phi_3045_: u32;
    var phi_3064_: bool;
    var phi_3080_: u32;
    var phi_3098_: bool;
    var phi_3114_: u32;
    var phi_3294_: bool;
    var phi_3273_: bool;
    var phi_3257_: bool;
    var phi_3279_: f32;
    var phi_3237_: bool;
    var phi_3280_: f32;
    var phi_3193_: bool;
    var phi_3199_: f32;
    var phi_3201_: f32;
    var phi_3202_: bool;
    var phi_3216_: bool;
    var phi_3222_: f32;
    var phi_3281_: f32;
    var phi_3300_: f32;
    var phi_3311_: u32;
    var phi_3025_: u32;
    var phi_3858_: bool;
    var phi_150_: f32;
    var phi_151_: bool;
    var phi_174_: f32;
    var phi_175_: bool;
    var phi_204_: u32;
    var phi_207_: f32;
    var phi_209_: f32;
    var phi_211_: f32;
    var phi_229_: u32;
    var phi_247_: bool;
    var phi_263_: u32;
    var phi_281_: bool;
    var phi_299_: u32;
    var phi_331_: bool;
    var phi_348_: bool;
    var phi_205_: u32;
    var phi_208_: f32;
    var phi_210_: f32;
    var phi_212_: f32;
    var phi_3903_: bool;
    var local: f32;
    var phi_371_: bool;
    var phi_401_: u32;
    var phi_402_: u32;
    var phi_3928_: bool;
    var phi_449_: bool;
    var local_1_: f32;
    var phi_465_: bool;
    var local_2_: f32;
    var phi_481_: bool;
    var phi_519_: u32;
    var phi_520_: u32;
    var phi_3937_: bool;
    var phi_580_: bool;
    var phi_598_: bool;
    var phi_610_: u32;
    var phi_613_: u32;
    var phi_611_: u32;
    var phi_614_: u32;
    var phi_625_: u32;
    var phi_628_: u32;
    var phi_630_: u32;
    var local_3_: u32;
    var phi_643_: u32;
    var phi_626_: u32;
    var phi_629_: u32;
    var phi_631_: u32;
    var phi_646_: u32;
    var phi_672_: bool;
    var phi_674_: bool;
    var phi_675_: bool;
    var phi_676_: bool;
    var phi_677_: bool;
    var phi_678_: bool;
    var phi_679_: bool;
    var phi_685_: u32;
    var phi_709_: u32;
    var phi_727_: bool;
    var phi_743_: u32;
    var phi_761_: bool;
    var phi_779_: u32;
    var phi_811_: bool;
    var phi_828_: bool;
    var phi_835_: vec4<f32>;
    var phi_846_: u32;
    var phi_686_: u32;
    var phi_3950_: bool;
    var phi_3989_: bool;
    var phi_3988_: bool;
    var phi_855_: u32;
    var phi_874_: u32;
    var phi_892_: bool;
    var phi_908_: u32;
    var phi_856_: u32;
    var phi_3985_: bool;
    var phi_943_: u32;
    var phi_944_: u32;
    var phi_945_: bool;
    var phi_3388_: bool;
    var phi_1036_: bool;
    var phi_3448_: bool;
    var phi_1001_: bool;
    var phi_1042_: f32;
    var phi_1043_: f32;
    var phi_1044_: f32;
    var phi_1045_: bool;
    var phi_4005_: bool;
    var phi_1052_: type_30_;
    var phi_1055_: f32;
    var phi_1057_: bool;
    var phi_1053_: type_30_;
    var phi_1080_: type_30_;
    var phi_1097_: u32;
    var phi_1100_: u32;
    var phi_1098_: u32;
    var phi_1101_: u32;
    var phi_1112_: u32;
    var phi_1115_: u32;
    var phi_1117_: u32;
    var local_4_: u32;
    var phi_1130_: u32;
    var phi_1113_: u32;
    var phi_1116_: u32;
    var phi_1118_: u32;
    var phi_1133_: u32;
    var phi_1135_: u32;
    var phi_1138_: f32;
    var phi_1140_: f32;
    var phi_1142_: f32;
    var phi_1162_: u32;
    var phi_1193_: u32;
    var phi_1210_: f32;
    var phi_1136_: u32;
    var phi_1139_: f32;
    var phi_1141_: f32;
    var phi_1143_: f32;
    var phi_3996_: bool;
    var local_5_: f32;
    var phi_1222_: bool;
    var local_6_: f32;
    var phi_1238_: bool;
    var phi_1276_: u32;
    var phi_1277_: u32;
    var phi_4030_: bool;
    var phi_1337_: bool;
    var phi_1355_: bool;
    var local_7_: f32;
    var phi_1374_: bool;
    var phi_1404_: u32;
    var phi_1405_: u32;
    var phi_4041_: bool;
    var phi_1452_: bool;
    var phi_4067_: bool;
    var phi_1458_: f32;
    var phi_3508_: bool;
    var phi_1554_: bool;
    var phi_3568_: bool;
    var phi_1519_: bool;
    var phi_1560_: f32;
    var phi_1561_: f32;
    var phi_1562_: f32;
    var phi_1563_: bool;
    var phi_1579_: bool;
    var phi_3628_: bool;
    var phi_1679_: bool;
    var phi_3688_: bool;
    var phi_1644_: bool;
    var phi_1685_: f32;
    var phi_1686_: f32;
    var phi_1687_: f32;
    var phi_1688_: bool;
    var phi_1704_: bool;
    var phi_1710_: f32;
    var phi_1713_: u32;
    var phi_1734_: u32;
    var phi_1753_: u32;
    var phi_1773_: bool;
    var phi_1714_: u32;
    var phi_4054_: bool;
    var phi_4103_: bool;
    var phi_1786_: f32;
    var phi_1787_: f32;
    var phi_1788_: f32;
    var phi_1789_: bool;
    var phi_4102_: bool;
    var phi_1791_: f32;
    var phi_1792_: f32;
    var phi_1793_: f32;
    var phi_1794_: bool;
    var phi_1795_: bool;
    var phi_4099_: bool;
    var phi_1056_: f32;
    var phi_1058_: bool;
    var phi_4098_: bool;
    var phi_1828_: u32;
    var phi_1847_: u32;
    var local_8_: f32;
    var phi_1866_: bool;
    var phi_1829_: u32;
    var phi_4104_: bool;
    var local_9_: f32;
    var phi_4120_: bool;
    var phi_1888_: type_30_;
    var phi_1889_: type_30_;
    var phi_1912_: type_30_;
    var phi_1931_: u32;
    var phi_1934_: u32;
    var phi_1932_: u32;
    var phi_1935_: u32;
    var phi_1946_: u32;
    var phi_1949_: u32;
    var phi_1951_: u32;
    var local_10_: u32;
    var phi_1964_: u32;
    var phi_1947_: u32;
    var phi_1950_: u32;
    var phi_1952_: u32;
    var phi_1967_: u32;
    var phi_1969_: u32;
    var phi_1972_: f32;
    var phi_1992_: u32;
    var phi_2011_: u32;
    var phi_1970_: u32;
    var phi_1973_: f32;
    var phi_4111_: bool;
    var local_11_: f32;
    var phi_2032_: bool;
    var phi_2062_: u32;
    var phi_2063_: u32;
    var phi_4127_: bool;
    var phi_2110_: bool;
    var phi_2142_: bool;
    var phi_2150_: u32;
    var phi_2171_: u32;
    var phi_2190_: u32;
    var phi_2210_: bool;
    var phi_2151_: u32;
    var phi_4136_: bool;
    var phi_4150_: bool;
    var phi_4149_: bool;
    var phi_2224_: u32;
    var phi_2227_: f32;
    var phi_2229_: f32;
    var phi_2247_: u32;
    var phi_2265_: bool;
    var phi_2281_: u32;
    var phi_2296_: f32;
    var phi_2308_: bool;
    var phi_2225_: u32;
    var phi_2228_: f32;
    var phi_2230_: f32;
    var phi_4151_: bool;
    var local_12_: f32;
    var phi_2327_: bool;
    var local_13_: f32;
    var phi_2343_: bool;
    var phi_2381_: u32;
    var phi_2382_: u32;
    var phi_4163_: bool;
    var phi_2442_: bool;
    var phi_2460_: bool;
    var phi_2482_: bool;
    var phi_2483_: bool;
    var phi_2484_: bool;
    var phi_3748_: bool;
    var phi_2575_: bool;
    var phi_3808_: bool;
    var phi_2540_: bool;
    var phi_2581_: f32;
    var phi_2582_: f32;
    var phi_2583_: f32;
    var phi_2584_: bool;
    var phi_2595_: u32;
    var phi_2614_: u32;
    var phi_2632_: bool;
    var phi_2648_: u32;
    var phi_2666_: bool;
    var phi_2682_: u32;
    var phi_2703_: u32;
    var phi_2743_: bool;
    var phi_2726_: bool;
    var phi_2749_: f32;
    var phi_2763_: u32;
    var phi_2780_: u32;
    var phi_2956_: bool;
    var phi_2935_: bool;
    var phi_2919_: bool;
    var phi_2941_: f32;
    var phi_2899_: bool;
    var phi_2942_: f32;
    var phi_2855_: bool;
    var phi_2861_: f32;
    var phi_2863_: f32;
    var phi_2864_: bool;
    var phi_2878_: bool;
    var phi_2884_: f32;
    var phi_2943_: f32;
    var phi_2962_: f32;
    var phi_2973_: u32;
    var phi_2596_: u32;
    var phi_4174_: bool;
    var local_14_: u32;
    var local_15_: u32;
    var local_16_: u32;
    var local_17_: u32;
    var local_18_: u32;
    var local_19_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e53_ = wg_id_1_;
            let _e54_ = lid_1_;
            let _e55_ = sg_lid_1_;
            let _e57_ = arrayLength((&theta.member));
            let _e59_ = arrayLength((&grad.member));
            let _e61_ = arrayLength((&param_bounds.member));
            let _e63_ = arrayLength((&lbfgs_s_y.member));
            let _e65_ = arrayLength((&search_dir.member));
            if (_e53_.x == 0u) {
                let _e68_ = (_e54_ & 255u);
                let _e71_ = uniforms.member.member;
                let _e74_ = uniforms.member.member_3_;
                if (_e74_ == 0u) {
                    let _e267_ = uniforms.member.member_1_;
                    let _e269_ = select(_e267_, 8u, (8u < _e267_));
                    let _e272_ = uniforms.member.member_5_;
                    if ((bitcast<u32>(_e272_) & 2139095040u) != 2139095040u) {
                        let _e276_ = (_e272_ > 0f);
                        phi_150_ = select(f32(), _e272_, _e276_);
                        phi_151_ = select(true, false, _e276_);
                    } else {
                        phi_150_ = f32();
                        phi_151_ = true;
                    }
                    let _e280_ = phi_150_;
                    let _e282_ = phi_151_;
                    let _e283_ = select(_e280_, 0f, _e282_);
                    let _e286_ = uniforms.member.member_6_;
                    if ((bitcast<u32>(_e286_) & 2139095040u) != 2139095040u) {
                        let _e290_ = (_e286_ >= 0f);
                        phi_174_ = select(f32(), _e286_, _e290_);
                        phi_175_ = select(true, false, _e290_);
                    } else {
                        phi_174_ = f32();
                        phi_175_ = true;
                    }
                    let _e294_ = phi_174_;
                    let _e296_ = phi_175_;
                    let _e297_ = select(_e294_, 0f, _e296_);
                    let _e300_ = uniforms.member.member_4_;
                    let _e307_ = uniforms.member.member_2_;
                    if (_e307_ > 0u) {
                        if (_e269_ > 0u) {
                            phi_204_ = _e68_;
                            phi_207_ = 0f;
                            phi_209_ = 0f;
                            phi_211_ = 0f;
                            loop {
                                let _e311_ = phi_204_;
                                let _e313_ = phi_207_;
                                let _e315_ = phi_209_;
                                let _e317_ = phi_211_;
                                local = _e317_;
                                local_1_ = _e315_;
                                local_2_ = _e313_;
                                let _e318_ = (_e311_ < _e71_);
                                if _e318_ {
                                    if (_e57_ == 0u) {
                                        phi_229_ = 0u;
                                    } else {
                                        let _e320_ = (_e57_ - 1u);
                                        phi_229_ = select(_e320_, _e311_, (_e311_ < _e320_));
                                    }
                                    let _e324_ = phi_229_;
                                    if (_e324_ < _e57_) {
                                    } else {
                                        phi_3903_ = true;
                                        break;
                                    }
                                    let _e328_ = theta.member[_e324_];
                                    let _e329_ = bitcast<u32>(_e328_);
                                    let _e330_ = (_e329_ & 2139095040u);
                                    if (_e330_ == 2139095040u) {
                                        phi_247_ = true;
                                    } else {
                                        phi_247_ = (_e330_ == 0u);
                                    }
                                    let _e334_ = phi_247_;
                                    if (_e59_ == 0u) {
                                        phi_263_ = 0u;
                                    } else {
                                        let _e338_ = (_e59_ - 1u);
                                        phi_263_ = select(_e338_, _e311_, (_e311_ < _e338_));
                                    }
                                    let _e342_ = phi_263_;
                                    if (_e342_ < _e59_) {
                                    } else {
                                        phi_3903_ = true;
                                        break;
                                    }
                                    let _e346_ = grad.member[_e342_];
                                    let _e347_ = bitcast<u32>(_e346_);
                                    let _e348_ = (_e347_ & 2139095040u);
                                    if (_e348_ == 2139095040u) {
                                        phi_281_ = true;
                                    } else {
                                        phi_281_ = (_e348_ == 0u);
                                    }
                                    let _e352_ = phi_281_;
                                    let _e356_ = ((_e269_ * _e71_) + _e311_);
                                    if (_e63_ == 0u) {
                                        phi_299_ = 0u;
                                    } else {
                                        let _e358_ = (_e63_ - 1u);
                                        phi_299_ = select(_e358_, _e356_, (_e356_ < _e358_));
                                    }
                                    let _e362_ = phi_299_;
                                    if (_e362_ < _e63_) {
                                    } else {
                                        phi_3903_ = true;
                                        break;
                                    }
                                    let _e367_ = lbfgs_s_y.member[_e362_].x;
                                    let _e371_ = lbfgs_s_y.member[_e362_].y;
                                    let _e373_ = bitcast<u32>((bitcast<f32>(select(_e329_, 0u, _e334_)) - _e367_));
                                    let _e374_ = (_e373_ & 2139095040u);
                                    if (_e374_ == 2139095040u) {
                                        phi_331_ = true;
                                    } else {
                                        phi_331_ = (_e374_ == 0u);
                                    }
                                    let _e378_ = phi_331_;
                                    let _e380_ = bitcast<f32>(select(_e373_, 0u, _e378_));
                                    let _e382_ = bitcast<u32>((bitcast<f32>(select(_e347_, 0u, _e352_)) - _e371_));
                                    let _e383_ = (_e382_ & 2139095040u);
                                    if (_e383_ == 2139095040u) {
                                        phi_348_ = true;
                                    } else {
                                        phi_348_ = (_e383_ == 0u);
                                    }
                                    let _e387_ = phi_348_;
                                    let _e389_ = bitcast<f32>(select(_e382_, 0u, _e387_));
                                    phi_205_ = (_e311_ + 256u);
                                    phi_208_ = (_e313_ + (_e389_ * _e389_));
                                    phi_210_ = (_e315_ + (_e380_ * _e380_));
                                    phi_212_ = (_e317_ + (_e380_ * _e389_));
                                } else {
                                    phi_205_ = u32();
                                    phi_208_ = f32();
                                    phi_210_ = f32();
                                    phi_212_ = f32();
                                }
                                let _e398_ = phi_205_;
                                let _e400_ = phi_208_;
                                let _e402_ = phi_210_;
                                let _e404_ = phi_212_;
                                phi_204_ = _e398_;
                                phi_207_ = _e400_;
                                phi_209_ = _e402_;
                                phi_211_ = _e404_;
                                phi_3903_ = false;
                                if !(_e318_) {
                                    break;
                                }
                            }
                            let _e407_ = phi_3903_;
                            if _e407_ {
                                break;
                            }
                            let _e409_ = local;
                            let _e410_ = bitcast<u32>(_e409_);
                            let _e411_ = (_e410_ & 2139095040u);
                            if (_e411_ == 2139095040u) {
                                phi_371_ = true;
                            } else {
                                phi_371_ = (_e411_ == 0u);
                            }
                            let _e415_ = phi_371_;
                            let _e554 = subgroupAdd(bitcast<f32>(select(_e410_, 0u, _e415_)));
                            let _e419_ = (_e55_ == 0u);
                            let _e422_ = select(255u, _e68_, (_e68_ < 255u));
                            let _e423_ = (_e422_ < 256u);
                            if _e423_ {
                            } else {
                                break;
                            }
                            dot_scratch[_e422_] = select(0f, _e554, _e419_);
                            workgroupBarrier();
                            phi_401_ = 128u;
                            loop {
                                let _e426_ = phi_401_;
                                let _e427_ = (_e426_ > 0u);
                                if _e427_ {
                                    if (_e68_ < _e426_) {
                                        let _e429_ = (_e68_ < 256u);
                                        if _e429_ {
                                        } else {
                                            phi_3928_ = true;
                                            break;
                                        }
                                        let _e431_ = dot_scratch[_e68_];
                                        let _e432_ = (_e68_ + _e426_);
                                        if (_e432_ < 256u) {
                                        } else {
                                            phi_3928_ = true;
                                            break;
                                        }
                                        let _e435_ = dot_scratch[_e432_];
                                        if _e429_ {
                                        } else {
                                            phi_3928_ = true;
                                            break;
                                        }
                                        dot_scratch[_e68_] = (_e431_ + _e435_);
                                    }
                                    workgroupBarrier();
                                    phi_402_ = (_e426_ >> bitcast<u32>(1i));
                                } else {
                                    phi_402_ = u32();
                                }
                                let _e440_ = phi_402_;
                                phi_401_ = _e440_;
                                phi_3928_ = _e407_;
                                if !(_e427_) {
                                    break;
                                }
                            }
                            let _e443_ = phi_3928_;
                            if _e443_ {
                                break;
                            }
                            let _e445_ = dot_scratch[0];
                            let _e446_ = bitcast<u32>(_e445_);
                            let _e447_ = (_e446_ & 2139095040u);
                            if (_e447_ == 2139095040u) {
                                phi_449_ = true;
                            } else {
                                phi_449_ = (_e447_ == 0u);
                            }
                            let _e451_ = phi_449_;
                            let _e453_ = bitcast<f32>(select(_e446_, 0u, _e451_));
                            workgroupBarrier();
                            let _e455_ = local_1_;
                            let _e456_ = bitcast<u32>(_e455_);
                            let _e457_ = (_e456_ & 2139095040u);
                            if (_e457_ == 2139095040u) {
                                phi_465_ = true;
                            } else {
                                phi_465_ = (_e457_ == 0u);
                            }
                            let _e461_ = phi_465_;
                            let _e465_ = local_2_;
                            let _e466_ = bitcast<u32>(_e465_);
                            let _e467_ = (_e466_ & 2139095040u);
                            if (_e467_ == 2139095040u) {
                                phi_481_ = true;
                            } else {
                                phi_481_ = (_e467_ == 0u);
                            }
                            let _e471_ = phi_481_;
                            let _e634 = subgroupAdd(bitcast<f32>(select(_e456_, 0u, _e461_)));
                            let _e638 = subgroupAdd(bitcast<f32>(select(_e466_, 0u, _e471_)));
                            if _e423_ {
                            } else {
                                break;
                            }
                            scratch_q[_e422_] = select(0f, _e634, _e419_);
                            if _e423_ {
                            } else {
                                break;
                            }
                            dot_scratch[_e422_] = select(0f, _e638, _e419_);
                            workgroupBarrier();
                            phi_519_ = 128u;
                            loop {
                                let _e480_ = phi_519_;
                                let _e481_ = (_e480_ > 0u);
                                if _e481_ {
                                    if (_e68_ < _e480_) {
                                        let _e483_ = (_e68_ + _e480_);
                                        let _e484_ = (_e483_ < 256u);
                                        if _e484_ {
                                        } else {
                                            phi_3937_ = true;
                                            break;
                                        }
                                        let _e486_ = scratch_q[_e483_];
                                        let _e487_ = (_e68_ < 256u);
                                        if _e487_ {
                                        } else {
                                            phi_3937_ = true;
                                            break;
                                        }
                                        let _e489_ = scratch_q[_e68_];
                                        scratch_q[_e68_] = (_e489_ + _e486_);
                                        if _e484_ {
                                        } else {
                                            phi_3937_ = true;
                                            break;
                                        }
                                        let _e492_ = dot_scratch[_e483_];
                                        if _e487_ {
                                        } else {
                                            phi_3937_ = true;
                                            break;
                                        }
                                        let _e494_ = dot_scratch[_e68_];
                                        dot_scratch[_e68_] = (_e494_ + _e492_);
                                    }
                                    workgroupBarrier();
                                    phi_520_ = (_e480_ >> bitcast<u32>(1i));
                                } else {
                                    phi_520_ = u32();
                                }
                                let _e499_ = phi_520_;
                                phi_519_ = _e499_;
                                phi_3937_ = _e443_;
                                if !(_e481_) {
                                    break;
                                }
                            }
                            let _e502_ = phi_3937_;
                            if _e502_ {
                                break;
                            }
                            let _e504_ = scratch_q[0];
                            let _e505_ = bitcast<u32>(_e504_);
                            let _e506_ = (_e505_ & 2139095040u);
                            if (_e506_ == 2139095040u) {
                                phi_580_ = true;
                            } else {
                                phi_580_ = (_e506_ == 0u);
                            }
                            let _e510_ = phi_580_;
                            let _e512_ = bitcast<f32>(select(_e505_, 0u, _e510_));
                            let _e513_ = dot_scratch[0];
                            let _e514_ = bitcast<u32>(_e513_);
                            let _e515_ = (_e514_ & 2139095040u);
                            if (_e515_ == 2139095040u) {
                                phi_598_ = true;
                            } else {
                                phi_598_ = (_e515_ == 0u);
                            }
                            let _e519_ = phi_598_;
                            let _e521_ = bitcast<f32>(select(_e514_, 0u, _e519_));
                            workgroupBarrier();
                            let _e522_ = (_e307_ - 1u);
                            if (_e269_ <= 1u) {
                                phi_646_ = 0u;
                            } else {
                                phi_610_ = 0u;
                                phi_613_ = _e269_;
                                loop {
                                    let _e525_ = phi_610_;
                                    let _e527_ = phi_613_;
                                    local_14_ = _e527_;
                                    local_3_ = _e525_;
                                    let _e530_ = (_e527_ <= (_e522_ >> bitcast<u32>(1i)));
                                    if _e530_ {
                                        phi_611_ = (_e525_ + 1u);
                                        phi_614_ = (_e527_ << bitcast<u32>(1i));
                                    } else {
                                        phi_611_ = u32();
                                        phi_614_ = u32();
                                    }
                                    let _e535_ = phi_611_;
                                    let _e537_ = phi_614_;
                                    phi_610_ = _e535_;
                                    phi_613_ = _e537_;
                                    if !(_e530_) {
                                        break;
                                    }
                                }
                                phi_625_ = 0u;
                                let _e2071_ = local_14_;
                                phi_628_ = _e2071_;
                                phi_630_ = _e522_;
                                loop {
                                    let _e540_ = phi_625_;
                                    let _e542_ = phi_628_;
                                    let _e544_ = phi_630_;
                                    local_15_ = _e544_;
                                    let _e546_ = local_3_;
                                    let _e547_ = (_e540_ <= _e546_);
                                    if _e547_ {
                                        if (_e544_ >= _e542_) {
                                            phi_643_ = (_e544_ - _e542_);
                                        } else {
                                            phi_643_ = _e544_;
                                        }
                                        let _e551_ = phi_643_;
                                        phi_626_ = (_e540_ + 1u);
                                        phi_629_ = (_e542_ >> bitcast<u32>(1i));
                                        phi_631_ = _e551_;
                                    } else {
                                        phi_626_ = u32();
                                        phi_629_ = u32();
                                        phi_631_ = u32();
                                    }
                                    let _e556_ = phi_626_;
                                    let _e558_ = phi_629_;
                                    let _e560_ = phi_631_;
                                    phi_625_ = _e556_;
                                    phi_628_ = _e558_;
                                    phi_630_ = _e560_;
                                    if !(_e547_) {
                                        break;
                                    }
                                }
                                let _e2080_ = local_15_;
                                phi_646_ = _e2080_;
                            }
                            let _e563_ = phi_646_;
                            if ((bitcast<u32>(_e453_) & 2139095040u) != 2139095040u) {
                                if ((bitcast<u32>(_e512_) & 2139095040u) != 2139095040u) {
                                    if ((bitcast<u32>(_e521_) & 2139095040u) != 2139095040u) {
                                        let _e575_ = (_e453_ > (0.000000000001f * (_e512_ + _e521_)));
                                        if _e575_ {
                                            phi_672_ = (_e453_ > 0f);
                                        } else {
                                            phi_672_ = bool();
                                        }
                                        let _e578_ = phi_672_;
                                        phi_674_ = _e578_;
                                        phi_675_ = select(true, false, _e575_);
                                    } else {
                                        phi_674_ = bool();
                                        phi_675_ = true;
                                    }
                                    let _e581_ = phi_674_;
                                    let _e583_ = phi_675_;
                                    phi_676_ = _e581_;
                                    phi_677_ = _e583_;
                                } else {
                                    phi_676_ = bool();
                                    phi_677_ = true;
                                }
                                let _e585_ = phi_676_;
                                let _e587_ = phi_677_;
                                phi_678_ = _e585_;
                                phi_679_ = _e587_;
                            } else {
                                phi_678_ = bool();
                                phi_679_ = true;
                            }
                            let _e589_ = phi_678_;
                            let _e591_ = phi_679_;
                            phi_685_ = _e68_;
                            loop {
                                let _e594_ = phi_685_;
                                let _e595_ = (_e594_ < _e71_);
                                if _e595_ {
                                    let _e597_ = ((_e563_ * _e71_) + _e594_);
                                    if select(_e589_, false, _e591_) {
                                        if (_e57_ == 0u) {
                                            phi_709_ = 0u;
                                        } else {
                                            let _e599_ = (_e57_ - 1u);
                                            phi_709_ = select(_e599_, _e594_, (_e594_ < _e599_));
                                        }
                                        let _e603_ = phi_709_;
                                        if (_e603_ < _e57_) {
                                        } else {
                                            phi_3950_ = true;
                                            break;
                                        }
                                        let _e607_ = theta.member[_e603_];
                                        let _e608_ = bitcast<u32>(_e607_);
                                        let _e609_ = (_e608_ & 2139095040u);
                                        if (_e609_ == 2139095040u) {
                                            phi_727_ = true;
                                        } else {
                                            phi_727_ = (_e609_ == 0u);
                                        }
                                        let _e613_ = phi_727_;
                                        if (_e59_ == 0u) {
                                            phi_743_ = 0u;
                                        } else {
                                            let _e617_ = (_e59_ - 1u);
                                            phi_743_ = select(_e617_, _e594_, (_e594_ < _e617_));
                                        }
                                        let _e621_ = phi_743_;
                                        if (_e621_ < _e59_) {
                                        } else {
                                            phi_3950_ = true;
                                            break;
                                        }
                                        let _e625_ = grad.member[_e621_];
                                        let _e626_ = bitcast<u32>(_e625_);
                                        let _e627_ = (_e626_ & 2139095040u);
                                        if (_e627_ == 2139095040u) {
                                            phi_761_ = true;
                                        } else {
                                            phi_761_ = (_e627_ == 0u);
                                        }
                                        let _e631_ = phi_761_;
                                        let _e635_ = ((_e269_ * _e71_) + _e594_);
                                        if (_e63_ == 0u) {
                                            phi_779_ = 0u;
                                        } else {
                                            let _e637_ = (_e63_ - 1u);
                                            phi_779_ = select(_e637_, _e635_, (_e635_ < _e637_));
                                        }
                                        let _e641_ = phi_779_;
                                        if (_e641_ < _e63_) {
                                        } else {
                                            phi_3950_ = true;
                                            break;
                                        }
                                        let _e646_ = lbfgs_s_y.member[_e641_].x;
                                        let _e650_ = lbfgs_s_y.member[_e641_].y;
                                        let _e652_ = bitcast<u32>((bitcast<f32>(select(_e608_, 0u, _e613_)) - _e646_));
                                        let _e653_ = (_e652_ & 2139095040u);
                                        if (_e653_ == 2139095040u) {
                                            phi_811_ = true;
                                        } else {
                                            phi_811_ = (_e653_ == 0u);
                                        }
                                        let _e657_ = phi_811_;
                                        let _e661_ = bitcast<u32>((bitcast<f32>(select(_e626_, 0u, _e631_)) - _e650_));
                                        let _e662_ = (_e661_ & 2139095040u);
                                        if (_e662_ == 2139095040u) {
                                            phi_828_ = true;
                                        } else {
                                            phi_828_ = (_e662_ == 0u);
                                        }
                                        let _e666_ = phi_828_;
                                        phi_835_ = vec4<f32>(bitcast<f32>(select(_e652_, 0u, _e657_)), bitcast<f32>(select(_e661_, 0u, _e666_)), 0f, 0f);
                                    } else {
                                        phi_835_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e671_ = phi_835_;
                                    if (_e63_ == 0u) {
                                        phi_846_ = 0u;
                                    } else {
                                        let _e673_ = (_e63_ - 1u);
                                        phi_846_ = select(_e673_, _e597_, (_e597_ < _e673_));
                                    }
                                    let _e677_ = phi_846_;
                                    if (_e677_ < _e63_) {
                                    } else {
                                        phi_3950_ = true;
                                        break;
                                    }
                                    lbfgs_s_y.member[_e677_] = _e671_;
                                    phi_686_ = (_e594_ + 256u);
                                } else {
                                    phi_686_ = u32();
                                }
                                let _e683_ = phi_686_;
                                phi_685_ = _e683_;
                                phi_3950_ = _e502_;
                                if !(_e595_) {
                                    break;
                                }
                            }
                            let _e686_ = phi_3950_;
                            if _e686_ {
                                break;
                            }
                            workgroupBarrier();
                            phi_3989_ = _e686_;
                        } else {
                            phi_3989_ = false;
                        }
                        let _e688_ = phi_3989_;
                        phi_3988_ = _e688_;
                    } else {
                        phi_3988_ = false;
                    }
                    let _e690_ = phi_3988_;
                    phi_855_ = _e68_;
                    loop {
                        let _e692_ = phi_855_;
                        let _e693_ = (_e692_ < _e71_);
                        if _e693_ {
                            if (_e59_ == 0u) {
                                phi_874_ = 0u;
                            } else {
                                let _e695_ = (_e59_ - 1u);
                                phi_874_ = select(_e695_, _e692_, (_e692_ < _e695_));
                            }
                            let _e699_ = phi_874_;
                            if (_e699_ < _e59_) {
                            } else {
                                phi_3985_ = true;
                                break;
                            }
                            let _e703_ = grad.member[_e699_];
                            let _e704_ = bitcast<u32>(_e703_);
                            let _e705_ = (_e704_ & 2139095040u);
                            if (_e705_ == 2139095040u) {
                                phi_892_ = true;
                            } else {
                                phi_892_ = (_e705_ == 0u);
                            }
                            let _e709_ = phi_892_;
                            if (_e65_ == 0u) {
                                phi_908_ = 0u;
                            } else {
                                let _e713_ = (_e65_ - 1u);
                                phi_908_ = select(_e713_, _e692_, (_e692_ < _e713_));
                            }
                            let _e717_ = phi_908_;
                            if (_e717_ < _e65_) {
                            } else {
                                phi_3985_ = true;
                                break;
                            }
                            search_dir.member[_e717_] = bitcast<f32>(select(_e704_, 0u, _e709_));
                            phi_856_ = (_e692_ + 256u);
                        } else {
                            phi_856_ = u32();
                        }
                        let _e723_ = phi_856_;
                        phi_855_ = _e723_;
                        phi_3985_ = _e690_;
                        if !(_e693_) {
                            break;
                        }
                    }
                    let _e726_ = phi_3985_;
                    if _e726_ {
                        break;
                    }
                    let _e727_ = (_e68_ < 32u);
                    if _e727_ {
                        if _e727_ {
                        } else {
                            break;
                        }
                        lbfgs_scalars[_e68_] = 0f;
                    }
                    workgroupBarrier();
                    if (_e307_ == 0u) {
                        phi_944_ = u32();
                        phi_945_ = true;
                    } else {
                        let _e730_ = (_e269_ == 0u);
                        if _e730_ {
                            phi_943_ = u32();
                        } else {
                            phi_943_ = select(_e307_, _e269_, (_e269_ < _e307_));
                        }
                        let _e734_ = phi_943_;
                        phi_944_ = _e734_;
                        phi_945_ = _e730_;
                    }
                    let _e736_ = phi_944_;
                    let _e738_ = phi_945_;
                    let _e739_ = select(_e736_, 0u, _e738_);
                    let _e740_ = (1f + _e283_);
                    let _e743_ = ((bitcast<u32>(_e740_) & 2139095040u) != 2139095040u);
                    if _e743_ {
                        let _e744_ = (_e740_ <= 0f);
                        if _e744_ {
                            phi_1043_ = f32();
                        } else {
                            if (_e740_ < 0.00001f) {
                                let _e776_ = (select(0.000000000000000000000001f, _e740_, (_e740_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e776_ != _e776_) {
                                    phi_3448_ = true;
                                } else {
                                    phi_3448_ = (0.000001f >= _e776_);
                                }
                                let _e780_ = phi_3448_;
                                let _e781_ = select(_e776_, 0.000001f, _e780_);
                                let _e784_ = bitcast<f32>((2129859011u - bitcast<u32>(_e781_)));
                                let _e788_ = (_e784_ + (_e784_ * (1f - (_e781_ * _e784_))));
                                let _e792_ = (_e788_ + (_e788_ * (1f - (_e781_ * _e788_))));
                                let _e798_ = bitcast<u32>(((_e792_ + (_e792_ * (1f - (_e781_ * _e792_)))) * 18446744000000000000f));
                                let _e799_ = (_e798_ & 2139095040u);
                                if (_e799_ == 2139095040u) {
                                    phi_1001_ = true;
                                } else {
                                    phi_1001_ = (_e799_ == 0u);
                                }
                                let _e803_ = phi_1001_;
                                phi_1042_ = bitcast<f32>(select(_e798_, 0u, _e803_));
                            } else {
                                if (_e740_ != _e740_) {
                                    phi_3388_ = true;
                                } else {
                                    phi_3388_ = (0.000001f >= _e740_);
                                }
                                let _e749_ = phi_3388_;
                                let _e750_ = select(_e740_, 0.000001f, _e749_);
                                let _e753_ = bitcast<f32>((2129859011u - bitcast<u32>(_e750_)));
                                let _e757_ = (_e753_ + (_e753_ * (1f - (_e750_ * _e753_))));
                                let _e761_ = (_e757_ + (_e757_ * (1f - (_e750_ * _e757_))));
                                let _e766_ = bitcast<u32>((_e761_ + (_e761_ * (1f - (_e750_ * _e761_)))));
                                let _e767_ = (_e766_ & 2139095040u);
                                if (_e767_ == 2139095040u) {
                                    phi_1036_ = true;
                                } else {
                                    phi_1036_ = (_e767_ == 0u);
                                }
                                let _e771_ = phi_1036_;
                                phi_1042_ = bitcast<f32>(select(_e766_, 0u, _e771_));
                            }
                            let _e807_ = phi_1042_;
                            phi_1043_ = _e807_;
                        }
                        let _e809_ = phi_1043_;
                        phi_1044_ = _e809_;
                        phi_1045_ = _e744_;
                    } else {
                        phi_1044_ = f32();
                        phi_1045_ = true;
                    }
                    let _e811_ = phi_1044_;
                    let _e813_ = phi_1045_;
                    let _e815_ = type_30_(0u, _e739_);
                    phi_4005_ = _e726_;
                    phi_1052_ = _e815_;
                    phi_1055_ = select(_e811_, 0f, _e813_);
                    phi_1057_ = false;
                    loop {
                        let _e817_ = phi_4005_;
                        let _e819_ = phi_1052_;
                        let _e821_ = phi_1055_;
                        let _e823_ = phi_1057_;
                        local_8_ = _e821_;
                        local_9_ = _e821_;
                        if (_e819_.member < _e819_.member_1_) {
                            phi_1053_ = type_30_((_e819_.member + 1u), _e819_.member_1_);
                            phi_1080_ = type_30_(1u, _e819_.member);
                        } else {
                            phi_1053_ = _e819_;
                            phi_1080_ = type_30_(0u, u32());
                        }
                        let _e836_ = phi_1053_;
                        let _e838_ = phi_1080_;
                        let _e842_ = (bitcast<i32>(_e838_.member) != 0i);
                        if _e842_ {
                            let _e844_ = ((_e307_ - 1u) - _e838_.member_1_);
                            if (_e269_ <= 1u) {
                                phi_1133_ = 0u;
                            } else {
                                phi_1097_ = 0u;
                                phi_1100_ = _e269_;
                                loop {
                                    let _e847_ = phi_1097_;
                                    let _e849_ = phi_1100_;
                                    local_16_ = _e849_;
                                    local_4_ = _e847_;
                                    let _e852_ = (_e849_ <= (_e844_ >> bitcast<u32>(1i)));
                                    if _e852_ {
                                        phi_1098_ = (_e847_ + 1u);
                                        phi_1101_ = (_e849_ << bitcast<u32>(1i));
                                    } else {
                                        phi_1098_ = u32();
                                        phi_1101_ = u32();
                                    }
                                    let _e857_ = phi_1098_;
                                    let _e859_ = phi_1101_;
                                    phi_1097_ = _e857_;
                                    phi_1100_ = _e859_;
                                    if !(_e852_) {
                                        break;
                                    }
                                }
                                phi_1112_ = 0u;
                                let _e2132_ = local_16_;
                                phi_1115_ = _e2132_;
                                phi_1117_ = _e844_;
                                loop {
                                    let _e862_ = phi_1112_;
                                    let _e864_ = phi_1115_;
                                    let _e866_ = phi_1117_;
                                    local_17_ = _e866_;
                                    let _e868_ = local_4_;
                                    let _e869_ = (_e862_ <= _e868_);
                                    if _e869_ {
                                        if (_e866_ >= _e864_) {
                                            phi_1130_ = (_e866_ - _e864_);
                                        } else {
                                            phi_1130_ = _e866_;
                                        }
                                        let _e873_ = phi_1130_;
                                        phi_1113_ = (_e862_ + 1u);
                                        phi_1116_ = (_e864_ >> bitcast<u32>(1i));
                                        phi_1118_ = _e873_;
                                    } else {
                                        phi_1113_ = u32();
                                        phi_1116_ = u32();
                                        phi_1118_ = u32();
                                    }
                                    let _e878_ = phi_1113_;
                                    let _e880_ = phi_1116_;
                                    let _e882_ = phi_1118_;
                                    phi_1112_ = _e878_;
                                    phi_1115_ = _e880_;
                                    phi_1117_ = _e882_;
                                    if !(_e869_) {
                                        break;
                                    }
                                }
                                let _e2141_ = local_17_;
                                phi_1133_ = _e2141_;
                            }
                            let _e885_ = phi_1133_;
                            phi_1135_ = _e68_;
                            phi_1138_ = 0f;
                            phi_1140_ = 0f;
                            phi_1142_ = 0f;
                            loop {
                                let _e887_ = phi_1135_;
                                let _e889_ = phi_1138_;
                                let _e891_ = phi_1140_;
                                let _e893_ = phi_1142_;
                                local_5_ = _e893_;
                                local_6_ = _e891_;
                                local_7_ = _e889_;
                                let _e894_ = (_e887_ < _e71_);
                                if _e894_ {
                                    let _e896_ = ((_e885_ * _e71_) + _e887_);
                                    if (_e63_ == 0u) {
                                        phi_1162_ = 0u;
                                    } else {
                                        let _e898_ = (_e63_ - 1u);
                                        phi_1162_ = select(_e898_, _e896_, (_e896_ < _e898_));
                                    }
                                    let _e902_ = phi_1162_;
                                    if (_e902_ < _e63_) {
                                    } else {
                                        phi_3996_ = true;
                                        break;
                                    }
                                    let _e907_ = lbfgs_s_y.member[_e902_].x;
                                    let _e911_ = lbfgs_s_y.member[_e902_].y;
                                    if (_e65_ == 0u) {
                                        phi_1193_ = 0u;
                                    } else {
                                        let _e913_ = (_e65_ - 1u);
                                        phi_1193_ = select(_e913_, _e887_, (_e887_ < _e913_));
                                    }
                                    let _e917_ = phi_1193_;
                                    if (_e917_ < _e65_) {
                                    } else {
                                        phi_3996_ = true;
                                        break;
                                    }
                                    let _e921_ = search_dir.member[_e917_];
                                    if _e823_ {
                                        phi_1210_ = _e889_;
                                    } else {
                                        phi_1210_ = (_e889_ + (_e911_ * _e911_));
                                    }
                                    let _e929_ = phi_1210_;
                                    phi_1136_ = (_e887_ + 256u);
                                    phi_1139_ = _e929_;
                                    phi_1141_ = (_e891_ + (_e907_ * _e921_));
                                    phi_1143_ = (_e893_ + (_e907_ * _e911_));
                                } else {
                                    phi_1136_ = u32();
                                    phi_1139_ = f32();
                                    phi_1141_ = f32();
                                    phi_1143_ = f32();
                                }
                                let _e932_ = phi_1136_;
                                let _e934_ = phi_1139_;
                                let _e936_ = phi_1141_;
                                let _e938_ = phi_1143_;
                                phi_1135_ = _e932_;
                                phi_1138_ = _e934_;
                                phi_1140_ = _e936_;
                                phi_1142_ = _e938_;
                                phi_3996_ = _e817_;
                                if !(_e894_) {
                                    break;
                                }
                            }
                            let _e941_ = phi_3996_;
                            phi_4098_ = _e941_;
                            if _e941_ {
                                break;
                            }
                            let _e943_ = local_5_;
                            let _e944_ = bitcast<u32>(_e943_);
                            let _e945_ = (_e944_ & 2139095040u);
                            if (_e945_ == 2139095040u) {
                                phi_1222_ = true;
                            } else {
                                phi_1222_ = (_e945_ == 0u);
                            }
                            let _e949_ = phi_1222_;
                            let _e953_ = local_6_;
                            let _e954_ = bitcast<u32>(_e953_);
                            let _e955_ = (_e954_ & 2139095040u);
                            if (_e955_ == 2139095040u) {
                                phi_1238_ = true;
                            } else {
                                phi_1238_ = (_e955_ == 0u);
                            }
                            let _e959_ = phi_1238_;
                            let _e1269 = subgroupAdd(bitcast<f32>(select(_e944_, 0u, _e949_)));
                            let _e1273 = subgroupAdd(bitcast<f32>(select(_e954_, 0u, _e959_)));
                            let _e964_ = (_e55_ == 0u);
                            let _e968_ = select(255u, _e68_, (_e68_ < 255u));
                            let _e969_ = (_e968_ < 256u);
                            if _e969_ {
                            } else {
                                phi_4098_ = true;
                                break;
                            }
                            scratch_q[_e968_] = select(0f, _e1269, _e964_);
                            if _e969_ {
                            } else {
                                phi_4098_ = true;
                                break;
                            }
                            dot_scratch[_e968_] = select(0f, _e1273, _e964_);
                            workgroupBarrier();
                            phi_1276_ = 128u;
                            loop {
                                let _e973_ = phi_1276_;
                                let _e974_ = (_e973_ > 0u);
                                if _e974_ {
                                    if (_e68_ < _e973_) {
                                        let _e976_ = (_e68_ + _e973_);
                                        let _e977_ = (_e976_ < 256u);
                                        if _e977_ {
                                        } else {
                                            phi_4030_ = true;
                                            break;
                                        }
                                        let _e979_ = scratch_q[_e976_];
                                        let _e980_ = (_e68_ < 256u);
                                        if _e980_ {
                                        } else {
                                            phi_4030_ = true;
                                            break;
                                        }
                                        let _e982_ = scratch_q[_e68_];
                                        scratch_q[_e68_] = (_e982_ + _e979_);
                                        if _e977_ {
                                        } else {
                                            phi_4030_ = true;
                                            break;
                                        }
                                        let _e985_ = dot_scratch[_e976_];
                                        if _e980_ {
                                        } else {
                                            phi_4030_ = true;
                                            break;
                                        }
                                        let _e987_ = dot_scratch[_e68_];
                                        dot_scratch[_e68_] = (_e987_ + _e985_);
                                    }
                                    workgroupBarrier();
                                    phi_1277_ = (_e973_ >> bitcast<u32>(1i));
                                } else {
                                    phi_1277_ = u32();
                                }
                                let _e992_ = phi_1277_;
                                phi_1276_ = _e992_;
                                phi_4030_ = _e941_;
                                if !(_e974_) {
                                    break;
                                }
                            }
                            let _e995_ = phi_4030_;
                            phi_4098_ = _e995_;
                            if _e995_ {
                                break;
                            }
                            let _e997_ = scratch_q[0];
                            let _e998_ = bitcast<u32>(_e997_);
                            let _e999_ = (_e998_ & 2139095040u);
                            if (_e999_ == 2139095040u) {
                                phi_1337_ = true;
                            } else {
                                phi_1337_ = (_e999_ == 0u);
                            }
                            let _e1003_ = phi_1337_;
                            let _e1005_ = bitcast<f32>(select(_e998_, 0u, _e1003_));
                            let _e1007_ = dot_scratch[0];
                            let _e1008_ = bitcast<u32>(_e1007_);
                            let _e1009_ = (_e1008_ & 2139095040u);
                            if (_e1009_ == 2139095040u) {
                                phi_1355_ = true;
                            } else {
                                phi_1355_ = (_e1009_ == 0u);
                            }
                            let _e1013_ = phi_1355_;
                            workgroupBarrier();
                            if _e823_ {
                                phi_4067_ = _e995_;
                                phi_1458_ = 0f;
                            } else {
                                let _e1017_ = local_7_;
                                let _e1018_ = bitcast<u32>(_e1017_);
                                let _e1019_ = (_e1018_ & 2139095040u);
                                if (_e1019_ == 2139095040u) {
                                    phi_1374_ = true;
                                } else {
                                    phi_1374_ = (_e1019_ == 0u);
                                }
                                let _e1023_ = phi_1374_;
                                let _e1372 = subgroupAdd(bitcast<f32>(select(_e1018_, 0u, _e1023_)));
                                if _e969_ {
                                } else {
                                    phi_4098_ = true;
                                    break;
                                }
                                dot_scratch[_e968_] = select(0f, _e1372, _e964_);
                                workgroupBarrier();
                                phi_1404_ = 128u;
                                loop {
                                    let _e1029_ = phi_1404_;
                                    let _e1030_ = (_e1029_ > 0u);
                                    if _e1030_ {
                                        if (_e68_ < _e1029_) {
                                            let _e1032_ = (_e68_ < 256u);
                                            if _e1032_ {
                                            } else {
                                                phi_4041_ = true;
                                                break;
                                            }
                                            let _e1034_ = dot_scratch[_e68_];
                                            let _e1035_ = (_e68_ + _e1029_);
                                            if (_e1035_ < 256u) {
                                            } else {
                                                phi_4041_ = true;
                                                break;
                                            }
                                            let _e1038_ = dot_scratch[_e1035_];
                                            if _e1032_ {
                                            } else {
                                                phi_4041_ = true;
                                                break;
                                            }
                                            dot_scratch[_e68_] = (_e1034_ + _e1038_);
                                        }
                                        workgroupBarrier();
                                        phi_1405_ = (_e1029_ >> bitcast<u32>(1i));
                                    } else {
                                        phi_1405_ = u32();
                                    }
                                    let _e1043_ = phi_1405_;
                                    phi_1404_ = _e1043_;
                                    phi_4041_ = _e995_;
                                    if !(_e1030_) {
                                        break;
                                    }
                                }
                                let _e1046_ = phi_4041_;
                                phi_4098_ = _e1046_;
                                if _e1046_ {
                                    break;
                                }
                                let _e1047_ = dot_scratch[0];
                                let _e1048_ = bitcast<u32>(_e1047_);
                                let _e1049_ = (_e1048_ & 2139095040u);
                                if (_e1049_ == 2139095040u) {
                                    phi_1452_ = true;
                                } else {
                                    phi_1452_ = (_e1049_ == 0u);
                                }
                                let _e1053_ = phi_1452_;
                                workgroupBarrier();
                                phi_4067_ = _e1046_;
                                phi_1458_ = bitcast<f32>(select(_e1048_, 0u, _e1053_));
                            }
                            let _e1057_ = phi_4067_;
                            let _e1059_ = phi_1458_;
                            let _e1062_ = ((bitcast<u32>(_e1005_) & 2139095040u) != 2139095040u);
                            if _e1062_ {
                                let _e1063_ = (_e1005_ > 0f);
                                if _e1063_ {
                                    if _e1062_ {
                                        let _e1064_ = (_e1005_ <= 0f);
                                        if _e1064_ {
                                            phi_1561_ = f32();
                                        } else {
                                            if (_e1005_ < 0.00001f) {
                                                let _e1096_ = (select(0.000000000000000000000001f, _e1005_, (_e1005_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                if (_e1096_ != _e1096_) {
                                                    phi_3568_ = true;
                                                } else {
                                                    phi_3568_ = (0.000001f >= _e1096_);
                                                }
                                                let _e1100_ = phi_3568_;
                                                let _e1101_ = select(_e1096_, 0.000001f, _e1100_);
                                                let _e1104_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1101_)));
                                                let _e1108_ = (_e1104_ + (_e1104_ * (1f - (_e1101_ * _e1104_))));
                                                let _e1112_ = (_e1108_ + (_e1108_ * (1f - (_e1101_ * _e1108_))));
                                                let _e1118_ = bitcast<u32>(((_e1112_ + (_e1112_ * (1f - (_e1101_ * _e1112_)))) * 18446744000000000000f));
                                                let _e1119_ = (_e1118_ & 2139095040u);
                                                if (_e1119_ == 2139095040u) {
                                                    phi_1519_ = true;
                                                } else {
                                                    phi_1519_ = (_e1119_ == 0u);
                                                }
                                                let _e1123_ = phi_1519_;
                                                phi_1560_ = bitcast<f32>(select(_e1118_, 0u, _e1123_));
                                            } else {
                                                if (_e1005_ != _e1005_) {
                                                    phi_3508_ = true;
                                                } else {
                                                    phi_3508_ = (0.000001f >= _e1005_);
                                                }
                                                let _e1069_ = phi_3508_;
                                                let _e1070_ = select(_e1005_, 0.000001f, _e1069_);
                                                let _e1073_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1070_)));
                                                let _e1077_ = (_e1073_ + (_e1073_ * (1f - (_e1070_ * _e1073_))));
                                                let _e1081_ = (_e1077_ + (_e1077_ * (1f - (_e1070_ * _e1077_))));
                                                let _e1086_ = bitcast<u32>((_e1081_ + (_e1081_ * (1f - (_e1070_ * _e1081_)))));
                                                let _e1087_ = (_e1086_ & 2139095040u);
                                                if (_e1087_ == 2139095040u) {
                                                    phi_1554_ = true;
                                                } else {
                                                    phi_1554_ = (_e1087_ == 0u);
                                                }
                                                let _e1091_ = phi_1554_;
                                                phi_1560_ = bitcast<f32>(select(_e1086_, 0u, _e1091_));
                                            }
                                            let _e1127_ = phi_1560_;
                                            phi_1561_ = _e1127_;
                                        }
                                        let _e1129_ = phi_1561_;
                                        phi_1562_ = _e1129_;
                                        phi_1563_ = _e1064_;
                                    } else {
                                        phi_1562_ = f32();
                                        phi_1563_ = true;
                                    }
                                    let _e1131_ = phi_1562_;
                                    let _e1133_ = phi_1563_;
                                    let _e1134_ = select(_e1131_, 0f, _e1133_);
                                    let _e1136_ = bitcast<u32>((_e1134_ * bitcast<f32>(select(_e1008_, 0u, _e1013_))));
                                    let _e1137_ = (_e1136_ & 2139095040u);
                                    if (_e1137_ == 2139095040u) {
                                        phi_1579_ = true;
                                    } else {
                                        phi_1579_ = (_e1137_ == 0u);
                                    }
                                    let _e1141_ = phi_1579_;
                                    let _e1143_ = bitcast<f32>(select(_e1136_, 0u, _e1141_));
                                    if _e823_ {
                                        phi_1710_ = _e821_;
                                    } else {
                                        let _e1146_ = (_e1059_ + select(0.000000000001f, _e283_, (_e283_ > 0.000000000001f)));
                                        if ((bitcast<u32>(_e1146_) & 2139095040u) != 2139095040u) {
                                            let _e1150_ = (_e1146_ <= 0f);
                                            if _e1150_ {
                                                phi_1686_ = f32();
                                            } else {
                                                if (_e1146_ < 0.00001f) {
                                                    let _e1182_ = (select(0.000000000000000000000001f, _e1146_, (_e1146_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                    if (_e1182_ != _e1182_) {
                                                        phi_3688_ = true;
                                                    } else {
                                                        phi_3688_ = (0.000001f >= _e1182_);
                                                    }
                                                    let _e1186_ = phi_3688_;
                                                    let _e1187_ = select(_e1182_, 0.000001f, _e1186_);
                                                    let _e1190_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1187_)));
                                                    let _e1194_ = (_e1190_ + (_e1190_ * (1f - (_e1187_ * _e1190_))));
                                                    let _e1198_ = (_e1194_ + (_e1194_ * (1f - (_e1187_ * _e1194_))));
                                                    let _e1204_ = bitcast<u32>(((_e1198_ + (_e1198_ * (1f - (_e1187_ * _e1198_)))) * 18446744000000000000f));
                                                    let _e1205_ = (_e1204_ & 2139095040u);
                                                    if (_e1205_ == 2139095040u) {
                                                        phi_1644_ = true;
                                                    } else {
                                                        phi_1644_ = (_e1205_ == 0u);
                                                    }
                                                    let _e1209_ = phi_1644_;
                                                    phi_1685_ = bitcast<f32>(select(_e1204_, 0u, _e1209_));
                                                } else {
                                                    if (_e1146_ != _e1146_) {
                                                        phi_3628_ = true;
                                                    } else {
                                                        phi_3628_ = (0.000001f >= _e1146_);
                                                    }
                                                    let _e1155_ = phi_3628_;
                                                    let _e1156_ = select(_e1146_, 0.000001f, _e1155_);
                                                    let _e1159_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1156_)));
                                                    let _e1163_ = (_e1159_ + (_e1159_ * (1f - (_e1156_ * _e1159_))));
                                                    let _e1167_ = (_e1163_ + (_e1163_ * (1f - (_e1156_ * _e1163_))));
                                                    let _e1172_ = bitcast<u32>((_e1167_ + (_e1167_ * (1f - (_e1156_ * _e1167_)))));
                                                    let _e1173_ = (_e1172_ & 2139095040u);
                                                    if (_e1173_ == 2139095040u) {
                                                        phi_1679_ = true;
                                                    } else {
                                                        phi_1679_ = (_e1173_ == 0u);
                                                    }
                                                    let _e1177_ = phi_1679_;
                                                    phi_1685_ = bitcast<f32>(select(_e1172_, 0u, _e1177_));
                                                }
                                                let _e1213_ = phi_1685_;
                                                phi_1686_ = _e1213_;
                                            }
                                            let _e1215_ = phi_1686_;
                                            phi_1687_ = _e1215_;
                                            phi_1688_ = _e1150_;
                                        } else {
                                            phi_1687_ = f32();
                                            phi_1688_ = true;
                                        }
                                        let _e1217_ = phi_1687_;
                                        let _e1219_ = phi_1688_;
                                        let _e1222_ = bitcast<u32>((_e1005_ * select(_e1217_, 0f, _e1219_)));
                                        let _e1223_ = (_e1222_ & 2139095040u);
                                        if (_e1223_ == 2139095040u) {
                                            phi_1704_ = true;
                                        } else {
                                            phi_1704_ = (_e1223_ == 0u);
                                        }
                                        let _e1227_ = phi_1704_;
                                        phi_1710_ = bitcast<f32>(select(_e1222_, 0u, _e1227_));
                                    }
                                    let _e1231_ = phi_1710_;
                                    phi_1713_ = _e68_;
                                    loop {
                                        let _e1234_ = phi_1713_;
                                        let _e1235_ = (_e1234_ < _e71_);
                                        if _e1235_ {
                                            let _e1237_ = ((_e885_ * _e71_) + _e1234_);
                                            if (_e63_ == 0u) {
                                                phi_1734_ = 0u;
                                            } else {
                                                let _e1239_ = (_e63_ - 1u);
                                                phi_1734_ = select(_e1239_, _e1237_, (_e1237_ < _e1239_));
                                            }
                                            let _e1243_ = phi_1734_;
                                            if (_e1243_ < _e63_) {
                                            } else {
                                                phi_4054_ = true;
                                                break;
                                            }
                                            let _e1248_ = lbfgs_s_y.member[_e1243_].y;
                                            if (_e65_ == 0u) {
                                                phi_1753_ = 0u;
                                            } else {
                                                let _e1250_ = (_e65_ - 1u);
                                                phi_1753_ = select(_e1250_, _e1234_, (_e1234_ < _e1250_));
                                            }
                                            let _e1254_ = phi_1753_;
                                            let _e1255_ = (_e1254_ < _e65_);
                                            if _e1255_ {
                                            } else {
                                                phi_4054_ = true;
                                                break;
                                            }
                                            let _e1258_ = search_dir.member[_e1254_];
                                            let _e1261_ = bitcast<u32>((_e1258_ - (_e1143_ * _e1248_)));
                                            let _e1262_ = (_e1261_ & 2139095040u);
                                            if (_e1262_ == 2139095040u) {
                                                phi_1773_ = true;
                                            } else {
                                                phi_1773_ = (_e1262_ == 0u);
                                            }
                                            let _e1266_ = phi_1773_;
                                            if _e1255_ {
                                            } else {
                                                phi_4054_ = true;
                                                break;
                                            }
                                            search_dir.member[_e1254_] = bitcast<f32>(select(_e1261_, 0u, _e1266_));
                                            phi_1714_ = (_e1234_ + 256u);
                                        } else {
                                            phi_1714_ = u32();
                                        }
                                        let _e1271_ = phi_1714_;
                                        phi_1713_ = _e1271_;
                                        phi_4054_ = _e1057_;
                                        if !(_e1235_) {
                                            break;
                                        }
                                    }
                                    let _e1274_ = phi_4054_;
                                    phi_4098_ = _e1274_;
                                    if _e1274_ {
                                        break;
                                    }
                                    phi_4103_ = _e1274_;
                                    phi_1786_ = _e1231_;
                                    phi_1787_ = _e1143_;
                                    phi_1788_ = _e1134_;
                                    phi_1789_ = select(true, _e823_, _e823_);
                                } else {
                                    phi_4103_ = _e1057_;
                                    phi_1786_ = f32();
                                    phi_1787_ = f32();
                                    phi_1788_ = f32();
                                    phi_1789_ = bool();
                                }
                                let _e1276_ = phi_4103_;
                                let _e1278_ = phi_1786_;
                                let _e1280_ = phi_1787_;
                                let _e1282_ = phi_1788_;
                                let _e1284_ = phi_1789_;
                                phi_4102_ = _e1276_;
                                phi_1791_ = _e1278_;
                                phi_1792_ = _e1280_;
                                phi_1793_ = _e1282_;
                                phi_1794_ = _e1284_;
                                phi_1795_ = select(true, false, _e1063_);
                            } else {
                                phi_4102_ = _e1057_;
                                phi_1791_ = f32();
                                phi_1792_ = f32();
                                phi_1793_ = f32();
                                phi_1794_ = bool();
                                phi_1795_ = true;
                            }
                            let _e1287_ = phi_4102_;
                            let _e1289_ = phi_1791_;
                            let _e1291_ = phi_1792_;
                            let _e1293_ = phi_1793_;
                            let _e1295_ = phi_1794_;
                            let _e1297_ = phi_1795_;
                            if (_e68_ == 0u) {
                                if (_e838_.member_1_ < 32u) {
                                } else {
                                    phi_4098_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e838_.member_1_] = select(_e1291_, 0f, _e1297_);
                                let _e1305_ = (8u + _e838_.member_1_);
                                if (_e1305_ < 32u) {
                                } else {
                                    phi_4098_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e1305_] = select(_e1293_, 0f, _e1297_);
                                let _e1308_ = (16u + _e838_.member_1_);
                                if (_e1308_ < 32u) {
                                } else {
                                    phi_4098_ = true;
                                    break;
                                }
                                lbfgs_scalars[_e1308_] = _e1005_;
                            }
                            workgroupBarrier();
                            phi_4099_ = _e1287_;
                            phi_1056_ = select(_e1289_, _e821_, _e1297_);
                            phi_1058_ = select(_e1295_, _e823_, _e1297_);
                        } else {
                            phi_4099_ = _e817_;
                            phi_1056_ = f32();
                            phi_1058_ = bool();
                        }
                        let _e1312_ = phi_4099_;
                        let _e1314_ = phi_1056_;
                        let _e1316_ = phi_1058_;
                        phi_4005_ = _e1312_;
                        phi_1052_ = _e836_;
                        phi_1055_ = _e1314_;
                        phi_1057_ = _e1316_;
                        phi_4098_ = _e1312_;
                        if !(_e842_) {
                            break;
                        }
                    }
                    let _e1319_ = phi_4098_;
                    if _e1319_ {
                        break;
                    }
                    phi_1828_ = _e68_;
                    loop {
                        let _e1321_ = phi_1828_;
                        let _e1322_ = (_e1321_ < _e71_);
                        if _e1322_ {
                            if (_e65_ == 0u) {
                                phi_1847_ = 0u;
                            } else {
                                let _e1324_ = (_e65_ - 1u);
                                phi_1847_ = select(_e1324_, _e1321_, (_e1321_ < _e1324_));
                            }
                            let _e1328_ = phi_1847_;
                            let _e1329_ = (_e1328_ < _e65_);
                            if _e1329_ {
                            } else {
                                phi_4104_ = true;
                                break;
                            }
                            let _e1332_ = search_dir.member[_e1328_];
                            let _e1334_ = local_8_;
                            let _e1336_ = bitcast<u32>((_e1332_ * _e1334_));
                            let _e1337_ = (_e1336_ & 2139095040u);
                            if (_e1337_ == 2139095040u) {
                                phi_1866_ = true;
                            } else {
                                phi_1866_ = (_e1337_ == 0u);
                            }
                            let _e1341_ = phi_1866_;
                            if _e1329_ {
                            } else {
                                phi_4104_ = true;
                                break;
                            }
                            search_dir.member[_e1328_] = bitcast<f32>(select(_e1336_, 0u, _e1341_));
                            phi_1829_ = (_e1321_ + 256u);
                        } else {
                            phi_1829_ = u32();
                        }
                        let _e1346_ = phi_1829_;
                        phi_1828_ = _e1346_;
                        phi_4104_ = _e1319_;
                        if !(_e1322_) {
                            break;
                        }
                    }
                    let _e1349_ = phi_4104_;
                    if _e1349_ {
                        break;
                    }
                    let _e1350_ = (_e68_ == 0u);
                    if _e1350_ {
                        let _e1353_ = local_9_;
                        lbfgs_scalars[24] = _e1353_;
                        lbfgs_scalars[26] = f32(_e739_);
                    }
                    workgroupBarrier();
                    phi_4120_ = _e1349_;
                    phi_1888_ = _e815_;
                    loop {
                        let _e1357_ = phi_4120_;
                        let _e1359_ = phi_1888_;
                        if (_e1359_.member < _e1359_.member_1_) {
                            phi_1889_ = type_30_((_e1359_.member + 1u), _e1359_.member_1_);
                            phi_1912_ = type_30_(1u, _e1359_.member);
                        } else {
                            phi_1889_ = _e1359_;
                            phi_1912_ = type_30_(0u, u32());
                        }
                        let _e1372_ = phi_1889_;
                        let _e1374_ = phi_1912_;
                        let _e1378_ = (bitcast<i32>(_e1374_.member) != 0i);
                        if _e1378_ {
                            let _e1380_ = ((_e739_ - 1u) - _e1374_.member_1_);
                            let _e1382_ = ((_e307_ - 1u) - _e1380_);
                            if (_e269_ <= 1u) {
                                phi_1967_ = 0u;
                            } else {
                                phi_1931_ = 0u;
                                phi_1934_ = _e269_;
                                loop {
                                    let _e1385_ = phi_1931_;
                                    let _e1387_ = phi_1934_;
                                    local_18_ = _e1387_;
                                    local_10_ = _e1385_;
                                    let _e1390_ = (_e1387_ <= (_e1382_ >> bitcast<u32>(1i)));
                                    if _e1390_ {
                                        phi_1932_ = (_e1385_ + 1u);
                                        phi_1935_ = (_e1387_ << bitcast<u32>(1i));
                                    } else {
                                        phi_1932_ = u32();
                                        phi_1935_ = u32();
                                    }
                                    let _e1395_ = phi_1932_;
                                    let _e1397_ = phi_1935_;
                                    phi_1931_ = _e1395_;
                                    phi_1934_ = _e1397_;
                                    if !(_e1390_) {
                                        break;
                                    }
                                }
                                phi_1946_ = 0u;
                                let _e2229_ = local_18_;
                                phi_1949_ = _e2229_;
                                phi_1951_ = _e1382_;
                                loop {
                                    let _e1400_ = phi_1946_;
                                    let _e1402_ = phi_1949_;
                                    let _e1404_ = phi_1951_;
                                    local_19_ = _e1404_;
                                    let _e1406_ = local_10_;
                                    let _e1407_ = (_e1400_ <= _e1406_);
                                    if _e1407_ {
                                        if (_e1404_ >= _e1402_) {
                                            phi_1964_ = (_e1404_ - _e1402_);
                                        } else {
                                            phi_1964_ = _e1404_;
                                        }
                                        let _e1411_ = phi_1964_;
                                        phi_1947_ = (_e1400_ + 1u);
                                        phi_1950_ = (_e1402_ >> bitcast<u32>(1i));
                                        phi_1952_ = _e1411_;
                                    } else {
                                        phi_1947_ = u32();
                                        phi_1950_ = u32();
                                        phi_1952_ = u32();
                                    }
                                    let _e1416_ = phi_1947_;
                                    let _e1418_ = phi_1950_;
                                    let _e1420_ = phi_1952_;
                                    phi_1946_ = _e1416_;
                                    phi_1949_ = _e1418_;
                                    phi_1951_ = _e1420_;
                                    if !(_e1407_) {
                                        break;
                                    }
                                }
                                let _e2238_ = local_19_;
                                phi_1967_ = _e2238_;
                            }
                            let _e1423_ = phi_1967_;
                            phi_1969_ = _e68_;
                            phi_1972_ = 0f;
                            loop {
                                let _e1425_ = phi_1969_;
                                let _e1427_ = phi_1972_;
                                local_11_ = _e1427_;
                                let _e1428_ = (_e1425_ < _e71_);
                                if _e1428_ {
                                    let _e1430_ = ((_e1423_ * _e71_) + _e1425_);
                                    if (_e63_ == 0u) {
                                        phi_1992_ = 0u;
                                    } else {
                                        let _e1432_ = (_e63_ - 1u);
                                        phi_1992_ = select(_e1432_, _e1430_, (_e1430_ < _e1432_));
                                    }
                                    let _e1436_ = phi_1992_;
                                    if (_e1436_ < _e63_) {
                                    } else {
                                        phi_4111_ = true;
                                        break;
                                    }
                                    let _e1441_ = lbfgs_s_y.member[_e1436_].y;
                                    if (_e65_ == 0u) {
                                        phi_2011_ = 0u;
                                    } else {
                                        let _e1443_ = (_e65_ - 1u);
                                        phi_2011_ = select(_e1443_, _e1425_, (_e1425_ < _e1443_));
                                    }
                                    let _e1447_ = phi_2011_;
                                    if (_e1447_ < _e65_) {
                                    } else {
                                        phi_4111_ = true;
                                        break;
                                    }
                                    let _e1451_ = search_dir.member[_e1447_];
                                    phi_1970_ = (_e1425_ + 256u);
                                    phi_1973_ = (_e1427_ + (_e1441_ * _e1451_));
                                } else {
                                    phi_1970_ = u32();
                                    phi_1973_ = f32();
                                }
                                let _e1456_ = phi_1970_;
                                let _e1458_ = phi_1973_;
                                phi_1969_ = _e1456_;
                                phi_1972_ = _e1458_;
                                phi_4111_ = _e1357_;
                                if !(_e1428_) {
                                    break;
                                }
                            }
                            let _e1461_ = phi_4111_;
                            phi_4149_ = _e1461_;
                            if _e1461_ {
                                break;
                            }
                            let _e1463_ = local_11_;
                            let _e1464_ = bitcast<u32>(_e1463_);
                            let _e1465_ = (_e1464_ & 2139095040u);
                            if (_e1465_ == 2139095040u) {
                                phi_2032_ = true;
                            } else {
                                phi_2032_ = (_e1465_ == 0u);
                            }
                            let _e1469_ = phi_2032_;
                            let _e1964 = subgroupAdd(bitcast<f32>(select(_e1464_, 0u, _e1469_)));
                            let _e1476_ = select(255u, _e68_, (_e68_ < 255u));
                            if (_e1476_ < 256u) {
                            } else {
                                phi_4149_ = true;
                                break;
                            }
                            dot_scratch[_e1476_] = select(0f, _e1964, (_e55_ == 0u));
                            workgroupBarrier();
                            phi_2062_ = 128u;
                            loop {
                                let _e1480_ = phi_2062_;
                                let _e1481_ = (_e1480_ > 0u);
                                if _e1481_ {
                                    if (_e68_ < _e1480_) {
                                        let _e1483_ = (_e68_ < 256u);
                                        if _e1483_ {
                                        } else {
                                            phi_4127_ = true;
                                            break;
                                        }
                                        let _e1485_ = dot_scratch[_e68_];
                                        let _e1486_ = (_e68_ + _e1480_);
                                        if (_e1486_ < 256u) {
                                        } else {
                                            phi_4127_ = true;
                                            break;
                                        }
                                        let _e1489_ = dot_scratch[_e1486_];
                                        if _e1483_ {
                                        } else {
                                            phi_4127_ = true;
                                            break;
                                        }
                                        dot_scratch[_e68_] = (_e1485_ + _e1489_);
                                    }
                                    workgroupBarrier();
                                    phi_2063_ = (_e1480_ >> bitcast<u32>(1i));
                                } else {
                                    phi_2063_ = u32();
                                }
                                let _e1494_ = phi_2063_;
                                phi_2062_ = _e1494_;
                                phi_4127_ = _e1461_;
                                if !(_e1481_) {
                                    break;
                                }
                            }
                            let _e1497_ = phi_4127_;
                            phi_4149_ = _e1497_;
                            if _e1497_ {
                                break;
                            }
                            let _e1499_ = dot_scratch[0];
                            let _e1500_ = bitcast<u32>(_e1499_);
                            let _e1501_ = (_e1500_ & 2139095040u);
                            if (_e1501_ == 2139095040u) {
                                phi_2110_ = true;
                            } else {
                                phi_2110_ = (_e1501_ == 0u);
                            }
                            let _e1505_ = phi_2110_;
                            workgroupBarrier();
                            if (_e1380_ < 32u) {
                            } else {
                                phi_4149_ = true;
                                break;
                            }
                            let _e1510_ = lbfgs_scalars[_e1380_];
                            let _e1511_ = (8u + _e1380_);
                            if (_e1511_ < 32u) {
                            } else {
                                phi_4149_ = true;
                                break;
                            }
                            let _e1514_ = lbfgs_scalars[_e1511_];
                            let _e1516_ = bitcast<u32>((_e1514_ * bitcast<f32>(select(_e1500_, 0u, _e1505_))));
                            let _e1517_ = (_e1516_ & 2139095040u);
                            if (_e1517_ == 2139095040u) {
                                phi_2142_ = true;
                            } else {
                                phi_2142_ = (_e1517_ == 0u);
                            }
                            let _e1521_ = phi_2142_;
                            phi_2150_ = _e68_;
                            loop {
                                let _e1526_ = phi_2150_;
                                let _e1527_ = (_e1526_ < _e71_);
                                if _e1527_ {
                                    let _e1529_ = ((_e1423_ * _e71_) + _e1526_);
                                    if (_e63_ == 0u) {
                                        phi_2171_ = 0u;
                                    } else {
                                        let _e1531_ = (_e63_ - 1u);
                                        phi_2171_ = select(_e1531_, _e1529_, (_e1529_ < _e1531_));
                                    }
                                    let _e1535_ = phi_2171_;
                                    if (_e1535_ < _e63_) {
                                    } else {
                                        phi_4136_ = true;
                                        break;
                                    }
                                    let _e1540_ = lbfgs_s_y.member[_e1535_].x;
                                    if (_e65_ == 0u) {
                                        phi_2190_ = 0u;
                                    } else {
                                        let _e1542_ = (_e65_ - 1u);
                                        phi_2190_ = select(_e1542_, _e1526_, (_e1526_ < _e1542_));
                                    }
                                    let _e1546_ = phi_2190_;
                                    let _e1547_ = (_e1546_ < _e65_);
                                    if _e1547_ {
                                    } else {
                                        phi_4136_ = true;
                                        break;
                                    }
                                    let _e1550_ = search_dir.member[_e1546_];
                                    let _e1553_ = bitcast<u32>((_e1550_ + (_e1540_ * (_e1510_ - bitcast<f32>(select(_e1516_, 0u, _e1521_))))));
                                    let _e1554_ = (_e1553_ & 2139095040u);
                                    if (_e1554_ == 2139095040u) {
                                        phi_2210_ = true;
                                    } else {
                                        phi_2210_ = (_e1554_ == 0u);
                                    }
                                    let _e1558_ = phi_2210_;
                                    if _e1547_ {
                                    } else {
                                        phi_4136_ = true;
                                        break;
                                    }
                                    search_dir.member[_e1546_] = bitcast<f32>(select(_e1553_, 0u, _e1558_));
                                    phi_2151_ = (_e1526_ + 256u);
                                } else {
                                    phi_2151_ = u32();
                                }
                                let _e1563_ = phi_2151_;
                                phi_2150_ = _e1563_;
                                phi_4136_ = _e1497_;
                                if !(_e1527_) {
                                    break;
                                }
                            }
                            let _e1566_ = phi_4136_;
                            phi_4149_ = _e1566_;
                            if _e1566_ {
                                break;
                            }
                            workgroupBarrier();
                            phi_4150_ = _e1566_;
                        } else {
                            phi_4150_ = _e1357_;
                        }
                        let _e1568_ = phi_4150_;
                        phi_4120_ = _e1568_;
                        phi_1888_ = _e1372_;
                        phi_4149_ = _e1568_;
                        if !(_e1378_) {
                            break;
                        }
                    }
                    let _e1571_ = phi_4149_;
                    if _e1571_ {
                        break;
                    }
                    phi_2224_ = _e68_;
                    phi_2227_ = 0f;
                    phi_2229_ = 0f;
                    loop {
                        let _e1573_ = phi_2224_;
                        let _e1575_ = phi_2227_;
                        let _e1577_ = phi_2229_;
                        local_12_ = _e1577_;
                        local_13_ = _e1575_;
                        let _e1578_ = (_e1573_ < _e71_);
                        if _e1578_ {
                            if (_e59_ == 0u) {
                                phi_2247_ = 0u;
                            } else {
                                let _e1580_ = (_e59_ - 1u);
                                phi_2247_ = select(_e1580_, _e1573_, (_e1573_ < _e1580_));
                            }
                            let _e1584_ = phi_2247_;
                            if (_e1584_ < _e59_) {
                            } else {
                                phi_4151_ = true;
                                break;
                            }
                            let _e1588_ = grad.member[_e1584_];
                            let _e1589_ = bitcast<u32>(_e1588_);
                            let _e1590_ = (_e1589_ & 2139095040u);
                            if (_e1590_ == 2139095040u) {
                                phi_2265_ = true;
                            } else {
                                phi_2265_ = (_e1590_ == 0u);
                            }
                            let _e1594_ = phi_2265_;
                            if (_e65_ == 0u) {
                                phi_2281_ = 0u;
                            } else {
                                let _e1598_ = (_e65_ - 1u);
                                phi_2281_ = select(_e1598_, _e1573_, (_e1573_ < _e1598_));
                            }
                            let _e1602_ = phi_2281_;
                            if (_e1602_ < _e65_) {
                            } else {
                                phi_4151_ = true;
                                break;
                            }
                            let _e1606_ = search_dir.member[_e1602_];
                            if ((bitcast<u32>(_e1606_) & 2139095040u) != 2139095040u) {
                                phi_2296_ = _e1575_;
                            } else {
                                phi_2296_ = (_e1575_ + 1f);
                            }
                            let _e1612_ = phi_2296_;
                            let _e1614_ = bitcast<u32>(-(_e1606_));
                            let _e1615_ = (_e1614_ & 2139095040u);
                            if (_e1615_ == 2139095040u) {
                                phi_2308_ = true;
                            } else {
                                phi_2308_ = (_e1615_ == 0u);
                            }
                            let _e1619_ = phi_2308_;
                            phi_2225_ = (_e1573_ + 256u);
                            phi_2228_ = _e1612_;
                            phi_2230_ = (_e1577_ + (bitcast<f32>(select(_e1589_, 0u, _e1594_)) * bitcast<f32>(select(_e1614_, 0u, _e1619_))));
                        } else {
                            phi_2225_ = u32();
                            phi_2228_ = f32();
                            phi_2230_ = f32();
                        }
                        let _e1626_ = phi_2225_;
                        let _e1628_ = phi_2228_;
                        let _e1630_ = phi_2230_;
                        phi_2224_ = _e1626_;
                        phi_2227_ = _e1628_;
                        phi_2229_ = _e1630_;
                        phi_4151_ = _e1571_;
                        if !(_e1578_) {
                            break;
                        }
                    }
                    let _e1633_ = phi_4151_;
                    if _e1633_ {
                        break;
                    }
                    let _e1635_ = local_12_;
                    let _e1636_ = bitcast<u32>(_e1635_);
                    let _e1637_ = (_e1636_ & 2139095040u);
                    if (_e1637_ == 2139095040u) {
                        phi_2327_ = true;
                    } else {
                        phi_2327_ = (_e1637_ == 0u);
                    }
                    let _e1641_ = phi_2327_;
                    let _e1645_ = local_13_;
                    let _e1646_ = bitcast<u32>(_e1645_);
                    let _e1647_ = (_e1646_ & 2139095040u);
                    if (_e1647_ == 2139095040u) {
                        phi_2343_ = true;
                    } else {
                        phi_2343_ = (_e1647_ == 0u);
                    }
                    let _e1651_ = phi_2343_;
                    let _e2212 = subgroupAdd(bitcast<f32>(select(_e1636_, 0u, _e1641_)));
                    let _e2216 = subgroupAdd(bitcast<f32>(select(_e1646_, 0u, _e1651_)));
                    let _e1656_ = (_e55_ == 0u);
                    let _e1660_ = select(255u, _e68_, (_e68_ < 255u));
                    let _e1661_ = (_e1660_ < 256u);
                    if _e1661_ {
                    } else {
                        break;
                    }
                    scratch_q[_e1660_] = select(0f, _e2212, _e1656_);
                    if _e1661_ {
                    } else {
                        break;
                    }
                    dot_scratch[_e1660_] = select(0f, _e2216, _e1656_);
                    workgroupBarrier();
                    phi_2381_ = 128u;
                    loop {
                        let _e1665_ = phi_2381_;
                        let _e1666_ = (_e1665_ > 0u);
                        if _e1666_ {
                            if (_e68_ < _e1665_) {
                                let _e1668_ = (_e68_ + _e1665_);
                                let _e1669_ = (_e1668_ < 256u);
                                if _e1669_ {
                                } else {
                                    phi_4163_ = true;
                                    break;
                                }
                                let _e1671_ = scratch_q[_e1668_];
                                let _e1672_ = (_e68_ < 256u);
                                if _e1672_ {
                                } else {
                                    phi_4163_ = true;
                                    break;
                                }
                                let _e1674_ = scratch_q[_e68_];
                                scratch_q[_e68_] = (_e1674_ + _e1671_);
                                if _e1669_ {
                                } else {
                                    phi_4163_ = true;
                                    break;
                                }
                                let _e1677_ = dot_scratch[_e1668_];
                                if _e1672_ {
                                } else {
                                    phi_4163_ = true;
                                    break;
                                }
                                let _e1679_ = dot_scratch[_e68_];
                                dot_scratch[_e68_] = (_e1679_ + _e1677_);
                            }
                            workgroupBarrier();
                            phi_2382_ = (_e1665_ >> bitcast<u32>(1i));
                        } else {
                            phi_2382_ = u32();
                        }
                        let _e1684_ = phi_2382_;
                        phi_2381_ = _e1684_;
                        phi_4163_ = _e1633_;
                        if !(_e1666_) {
                            break;
                        }
                    }
                    let _e1687_ = phi_4163_;
                    if _e1687_ {
                        break;
                    }
                    let _e1689_ = scratch_q[0];
                    let _e1690_ = bitcast<u32>(_e1689_);
                    let _e1691_ = (_e1690_ & 2139095040u);
                    if (_e1691_ == 2139095040u) {
                        phi_2442_ = true;
                    } else {
                        phi_2442_ = (_e1691_ == 0u);
                    }
                    let _e1695_ = phi_2442_;
                    let _e1697_ = bitcast<f32>(select(_e1690_, 0u, _e1695_));
                    let _e1699_ = dot_scratch[0];
                    let _e1700_ = bitcast<u32>(_e1699_);
                    let _e1701_ = (_e1700_ & 2139095040u);
                    if (_e1701_ == 2139095040u) {
                        phi_2460_ = true;
                    } else {
                        phi_2460_ = (_e1701_ == 0u);
                    }
                    let _e1705_ = phi_2460_;
                    workgroupBarrier();
                    if _e1350_ {
                        lbfgs_scalars[25] = _e1697_;
                    }
                    workgroupBarrier();
                    if ((bitcast<u32>(_e1697_) & 2139095040u) != 2139095040u) {
                        let _e1712_ = (_e1697_ >= 0f);
                        if _e1712_ {
                            phi_2482_ = bool();
                        } else {
                            phi_2482_ = (bitcast<f32>(select(_e1700_, 0u, _e1705_)) > 0f);
                        }
                        let _e1715_ = phi_2482_;
                        phi_2483_ = _e1715_;
                        phi_2484_ = _e1712_;
                    } else {
                        phi_2483_ = bool();
                        phi_2484_ = true;
                    }
                    let _e1717_ = phi_2483_;
                    let _e1719_ = phi_2484_;
                    if _e743_ {
                        let _e1721_ = (_e740_ <= 0f);
                        if _e1721_ {
                            phi_2582_ = f32();
                        } else {
                            if (_e740_ < 0.00001f) {
                                let _e1753_ = (select(0.000000000000000000000001f, _e740_, (_e740_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e1753_ != _e1753_) {
                                    phi_3808_ = true;
                                } else {
                                    phi_3808_ = (0.000001f >= _e1753_);
                                }
                                let _e1757_ = phi_3808_;
                                let _e1758_ = select(_e1753_, 0.000001f, _e1757_);
                                let _e1761_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1758_)));
                                let _e1765_ = (_e1761_ + (_e1761_ * (1f - (_e1758_ * _e1761_))));
                                let _e1769_ = (_e1765_ + (_e1765_ * (1f - (_e1758_ * _e1765_))));
                                let _e1775_ = bitcast<u32>(((_e1769_ + (_e1769_ * (1f - (_e1758_ * _e1769_)))) * 18446744000000000000f));
                                let _e1776_ = (_e1775_ & 2139095040u);
                                if (_e1776_ == 2139095040u) {
                                    phi_2540_ = true;
                                } else {
                                    phi_2540_ = (_e1776_ == 0u);
                                }
                                let _e1780_ = phi_2540_;
                                phi_2581_ = bitcast<f32>(select(_e1775_, 0u, _e1780_));
                            } else {
                                if (_e740_ != _e740_) {
                                    phi_3748_ = true;
                                } else {
                                    phi_3748_ = (0.000001f >= _e740_);
                                }
                                let _e1726_ = phi_3748_;
                                let _e1727_ = select(_e740_, 0.000001f, _e1726_);
                                let _e1730_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1727_)));
                                let _e1734_ = (_e1730_ + (_e1730_ * (1f - (_e1727_ * _e1730_))));
                                let _e1738_ = (_e1734_ + (_e1734_ * (1f - (_e1727_ * _e1734_))));
                                let _e1743_ = bitcast<u32>((_e1738_ + (_e1738_ * (1f - (_e1727_ * _e1738_)))));
                                let _e1744_ = (_e1743_ & 2139095040u);
                                if (_e1744_ == 2139095040u) {
                                    phi_2575_ = true;
                                } else {
                                    phi_2575_ = (_e1744_ == 0u);
                                }
                                let _e1748_ = phi_2575_;
                                phi_2581_ = bitcast<f32>(select(_e1743_, 0u, _e1748_));
                            }
                            let _e1784_ = phi_2581_;
                            phi_2582_ = _e1784_;
                        }
                        let _e1786_ = phi_2582_;
                        phi_2583_ = _e1786_;
                        phi_2584_ = _e1721_;
                    } else {
                        phi_2583_ = f32();
                        phi_2584_ = true;
                    }
                    let _e1788_ = phi_2583_;
                    let _e1790_ = phi_2584_;
                    phi_2595_ = _e68_;
                    loop {
                        let _e1795_ = phi_2595_;
                        let _e1796_ = (_e1795_ < _e71_);
                        if _e1796_ {
                            let _e1797_ = (_e57_ == 0u);
                            if _e1797_ {
                                phi_2614_ = 0u;
                            } else {
                                let _e1798_ = (_e57_ - 1u);
                                phi_2614_ = select(_e1798_, _e1795_, (_e1795_ < _e1798_));
                            }
                            let _e1802_ = phi_2614_;
                            if (_e1802_ < _e57_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            let _e1806_ = theta.member[_e1802_];
                            let _e1807_ = bitcast<u32>(_e1806_);
                            let _e1808_ = (_e1807_ & 2139095040u);
                            if (_e1808_ == 2139095040u) {
                                phi_2632_ = true;
                            } else {
                                phi_2632_ = (_e1808_ == 0u);
                            }
                            let _e1812_ = phi_2632_;
                            let _e1814_ = bitcast<f32>(select(_e1807_, 0u, _e1812_));
                            if (_e59_ == 0u) {
                                phi_2648_ = 0u;
                            } else {
                                let _e1816_ = (_e59_ - 1u);
                                phi_2648_ = select(_e1816_, _e1795_, (_e1795_ < _e1816_));
                            }
                            let _e1820_ = phi_2648_;
                            if (_e1820_ < _e59_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            let _e1824_ = grad.member[_e1820_];
                            let _e1825_ = bitcast<u32>(_e1824_);
                            let _e1826_ = (_e1825_ & 2139095040u);
                            if (_e1826_ == 2139095040u) {
                                phi_2666_ = true;
                            } else {
                                phi_2666_ = (_e1826_ == 0u);
                            }
                            let _e1830_ = phi_2666_;
                            let _e1832_ = bitcast<f32>(select(_e1825_, 0u, _e1830_));
                            if (_e61_ == 0u) {
                                phi_2682_ = 0u;
                            } else {
                                let _e1834_ = (_e61_ - 1u);
                                phi_2682_ = select(_e1834_, _e1795_, (_e1795_ < _e1834_));
                            }
                            let _e1838_ = phi_2682_;
                            if (_e1838_ < _e61_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            let _e1843_ = param_bounds.member[_e1838_].x;
                            let _e1847_ = param_bounds.member[_e1838_].y;
                            let _e1848_ = (_e65_ == 0u);
                            if _e1848_ {
                                phi_2703_ = 0u;
                            } else {
                                let _e1849_ = (_e65_ - 1u);
                                phi_2703_ = select(_e1849_, _e1795_, (_e1795_ < _e1849_));
                            }
                            let _e1853_ = phi_2703_;
                            if (_e1853_ < _e65_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            let _e1857_ = search_dir.member[_e1853_];
                            if select(_e1717_, true, _e1719_) {
                                let _e1869_ = bitcast<u32>((-(_e1832_) * select(_e1788_, 0f, _e1790_)));
                                let _e1870_ = (_e1869_ & 2139095040u);
                                if (_e1870_ == 2139095040u) {
                                    phi_2726_ = true;
                                } else {
                                    phi_2726_ = (_e1870_ == 0u);
                                }
                                let _e1874_ = phi_2726_;
                                phi_2749_ = bitcast<f32>(select(_e1869_, 0u, _e1874_));
                            } else {
                                let _e1859_ = bitcast<u32>(-(_e1857_));
                                let _e1860_ = (_e1859_ & 2139095040u);
                                if (_e1860_ == 2139095040u) {
                                    phi_2743_ = true;
                                } else {
                                    phi_2743_ = (_e1860_ == 0u);
                                }
                                let _e1864_ = phi_2743_;
                                phi_2749_ = bitcast<f32>(select(_e1859_, 0u, _e1864_));
                            }
                            let _e1878_ = phi_2749_;
                            let _e1880_ = ((_e269_ * _e71_) + _e1795_);
                            if (_e63_ == 0u) {
                                phi_2763_ = 0u;
                            } else {
                                let _e1883_ = (_e63_ - 1u);
                                phi_2763_ = select(_e1883_, _e1880_, (_e1880_ < _e1883_));
                            }
                            let _e1887_ = phi_2763_;
                            if (_e1887_ < _e63_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            lbfgs_s_y.member[_e1887_] = vec4<f32>(_e1814_, _e1832_, select(0f, 1f, (_e739_ > 0u)), 0f);
                            if _e1848_ {
                                phi_2780_ = 0u;
                            } else {
                                let _e1891_ = (_e65_ - 1u);
                                phi_2780_ = select(_e1891_, _e1795_, (_e1795_ < _e1891_));
                            }
                            let _e1895_ = phi_2780_;
                            if (_e1895_ < _e65_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            search_dir.member[_e1895_] = _e1878_;
                            let _e1900_ = (_e1814_ + (select(0f, _e300_, ((bitcast<u32>(_e300_) & 2139095040u) != 2139095040u)) * _e1878_));
                            let _e1904_ = select(-1000000f, _e1843_, ((bitcast<u32>(_e1843_) & 2139095040u) != 2139095040u));
                            let _e1908_ = select(1000000f, _e1847_, ((bitcast<u32>(_e1847_) & 2139095040u) != 2139095040u));
                            let _e1909_ = (_e1904_ <= _e1908_);
                            let _e1910_ = select(_e1908_, _e1904_, _e1909_);
                            let _e1911_ = select(_e1904_, _e1908_, _e1909_);
                            let _e1912_ = (_e1910_ + _e297_);
                            let _e1913_ = (_e1911_ - _e297_);
                            let _e1914_ = (_e1912_ <= _e1913_);
                            if _e1914_ {
                                let _e1925_ = select(_e1913_, _e1912_, _e1914_);
                                let _e1926_ = select(_e1912_, _e1913_, _e1914_);
                                let _e1927_ = bitcast<u32>(_e1900_);
                                let _e1928_ = (_e1927_ & 2139095040u);
                                let _e1929_ = (_e1928_ == 2139095040u);
                                if _e1929_ {
                                    if ((_e1927_ & 2147483647u) > 2139095040u) {
                                        phi_2863_ = f32();
                                        phi_2864_ = true;
                                    } else {
                                        let _e1960_ = ((_e1927_ & 2147483648u) == 0u);
                                        if _e1960_ {
                                            let _e1961_ = bitcast<u32>(_e1926_);
                                            let _e1962_ = (_e1961_ & 2139095040u);
                                            if (_e1962_ == 2139095040u) {
                                                phi_2855_ = true;
                                            } else {
                                                phi_2855_ = (_e1962_ == 0u);
                                            }
                                            let _e1966_ = phi_2855_;
                                            phi_2861_ = bitcast<f32>(select(_e1961_, 0u, _e1966_));
                                        } else {
                                            phi_2861_ = f32();
                                        }
                                        let _e1970_ = phi_2861_;
                                        phi_2863_ = _e1970_;
                                        phi_2864_ = select(true, false, _e1960_);
                                    }
                                    let _e1973_ = phi_2863_;
                                    let _e1975_ = phi_2864_;
                                    if _e1975_ {
                                        let _e1976_ = bitcast<u32>(_e1925_);
                                        let _e1977_ = (_e1976_ & 2139095040u);
                                        if (_e1977_ == 2139095040u) {
                                            phi_2878_ = true;
                                        } else {
                                            phi_2878_ = (_e1977_ == 0u);
                                        }
                                        let _e1981_ = phi_2878_;
                                        phi_2884_ = bitcast<f32>(select(_e1976_, 0u, _e1981_));
                                    } else {
                                        phi_2884_ = _e1973_;
                                    }
                                    let _e1985_ = phi_2884_;
                                    phi_2943_ = _e1985_;
                                } else {
                                    if (_e1900_ < _e1925_) {
                                        let _e1947_ = bitcast<u32>(_e1925_);
                                        let _e1948_ = (_e1947_ & 2139095040u);
                                        if (_e1948_ == 2139095040u) {
                                            phi_2899_ = true;
                                        } else {
                                            phi_2899_ = (_e1948_ == 0u);
                                        }
                                        let _e1952_ = phi_2899_;
                                        phi_2942_ = bitcast<f32>(select(_e1947_, 0u, _e1952_));
                                    } else {
                                        if (_e1900_ > _e1926_) {
                                            let _e1937_ = bitcast<u32>(_e1926_);
                                            let _e1938_ = (_e1937_ & 2139095040u);
                                            if (_e1938_ == 2139095040u) {
                                                phi_2919_ = true;
                                            } else {
                                                phi_2919_ = (_e1938_ == 0u);
                                            }
                                            let _e1942_ = phi_2919_;
                                            phi_2941_ = bitcast<f32>(select(_e1937_, 0u, _e1942_));
                                        } else {
                                            if _e1929_ {
                                                phi_2935_ = true;
                                            } else {
                                                phi_2935_ = (_e1928_ == 0u);
                                            }
                                            let _e1934_ = phi_2935_;
                                            phi_2941_ = bitcast<f32>(select(_e1927_, 0u, _e1934_));
                                        }
                                        let _e1946_ = phi_2941_;
                                        phi_2942_ = _e1946_;
                                    }
                                    let _e1956_ = phi_2942_;
                                    phi_2943_ = _e1956_;
                                }
                                let _e1987_ = phi_2943_;
                                phi_2962_ = _e1987_;
                            } else {
                                let _e1917_ = bitcast<u32>((0.5f * (_e1910_ + _e1911_)));
                                let _e1918_ = (_e1917_ & 2139095040u);
                                if (_e1918_ == 2139095040u) {
                                    phi_2956_ = true;
                                } else {
                                    phi_2956_ = (_e1918_ == 0u);
                                }
                                let _e1922_ = phi_2956_;
                                phi_2962_ = bitcast<f32>(select(_e1917_, 0u, _e1922_));
                            }
                            let _e1989_ = phi_2962_;
                            if _e1797_ {
                                phi_2973_ = 0u;
                            } else {
                                let _e1990_ = (_e57_ - 1u);
                                phi_2973_ = select(_e1990_, _e1795_, (_e1795_ < _e1990_));
                            }
                            let _e1994_ = phi_2973_;
                            if (_e1994_ < _e57_) {
                            } else {
                                phi_4174_ = true;
                                break;
                            }
                            theta.member[_e1994_] = _e1989_;
                            phi_2596_ = (_e1795_ + 256u);
                        } else {
                            phi_2596_ = u32();
                        }
                        let _e2000_ = phi_2596_;
                        phi_2595_ = _e2000_;
                        phi_4174_ = _e1687_;
                        if !(_e1796_) {
                            break;
                        }
                    }
                    let _e2003_ = phi_4174_;
                    if _e2003_ {
                        break;
                    }
                } else {
                    let _e78_ = uniforms.member.member_1_;
                    let _e83_ = uniforms.member.member_6_;
                    if ((bitcast<u32>(_e83_) & 2139095040u) != 2139095040u) {
                        let _e87_ = (_e83_ >= 0f);
                        phi_3006_ = select(f32(), _e83_, _e87_);
                        phi_3007_ = select(true, false, _e87_);
                    } else {
                        phi_3006_ = f32();
                        phi_3007_ = true;
                    }
                    let _e91_ = phi_3006_;
                    let _e93_ = phi_3007_;
                    let _e94_ = select(_e91_, 0f, _e93_);
                    let _e97_ = uniforms.member.member_4_;
                    phi_3024_ = _e68_;
                    loop {
                        let _e103_ = phi_3024_;
                        let _e104_ = (_e103_ < _e71_);
                        if _e104_ {
                            let _e106_ = ((select(_e78_, 8u, (8u < _e78_)) * _e71_) + _e103_);
                            if (_e63_ == 0u) {
                                phi_3045_ = 0u;
                            } else {
                                let _e108_ = (_e63_ - 1u);
                                phi_3045_ = select(_e108_, _e106_, (_e106_ < _e108_));
                            }
                            let _e112_ = phi_3045_;
                            if (_e112_ < _e63_) {
                            } else {
                                phi_3858_ = true;
                                break;
                            }
                            let _e117_ = lbfgs_s_y.member[_e112_].x;
                            let _e118_ = bitcast<u32>(_e117_);
                            let _e119_ = (_e118_ & 2139095040u);
                            if (_e119_ == 2139095040u) {
                                phi_3064_ = true;
                            } else {
                                phi_3064_ = (_e119_ == 0u);
                            }
                            let _e123_ = phi_3064_;
                            if (_e65_ == 0u) {
                                phi_3080_ = 0u;
                            } else {
                                let _e127_ = (_e65_ - 1u);
                                phi_3080_ = select(_e127_, _e103_, (_e103_ < _e127_));
                            }
                            let _e131_ = phi_3080_;
                            if (_e131_ < _e65_) {
                            } else {
                                phi_3858_ = true;
                                break;
                            }
                            let _e135_ = search_dir.member[_e131_];
                            let _e136_ = bitcast<u32>(_e135_);
                            let _e137_ = (_e136_ & 2139095040u);
                            if (_e137_ == 2139095040u) {
                                phi_3098_ = true;
                            } else {
                                phi_3098_ = (_e137_ == 0u);
                            }
                            let _e141_ = phi_3098_;
                            if (_e61_ == 0u) {
                                phi_3114_ = 0u;
                            } else {
                                let _e145_ = (_e61_ - 1u);
                                phi_3114_ = select(_e145_, _e103_, (_e103_ < _e145_));
                            }
                            let _e149_ = phi_3114_;
                            if (_e149_ < _e61_) {
                            } else {
                                phi_3858_ = true;
                                break;
                            }
                            let _e154_ = param_bounds.member[_e149_].x;
                            let _e158_ = param_bounds.member[_e149_].y;
                            let _e160_ = (bitcast<f32>(select(_e118_, 0u, _e123_)) + (select(0f, _e97_, ((bitcast<u32>(_e97_) & 2139095040u) != 2139095040u)) * bitcast<f32>(select(_e136_, 0u, _e141_))));
                            let _e164_ = select(-1000000f, _e154_, ((bitcast<u32>(_e154_) & 2139095040u) != 2139095040u));
                            let _e168_ = select(1000000f, _e158_, ((bitcast<u32>(_e158_) & 2139095040u) != 2139095040u));
                            let _e169_ = (_e164_ <= _e168_);
                            let _e170_ = select(_e168_, _e164_, _e169_);
                            let _e171_ = select(_e164_, _e168_, _e169_);
                            let _e172_ = (_e170_ + _e94_);
                            let _e173_ = (_e171_ - _e94_);
                            let _e174_ = (_e172_ <= _e173_);
                            if _e174_ {
                                let _e185_ = select(_e173_, _e172_, _e174_);
                                let _e186_ = select(_e172_, _e173_, _e174_);
                                let _e187_ = bitcast<u32>(_e160_);
                                let _e188_ = (_e187_ & 2139095040u);
                                let _e189_ = (_e188_ == 2139095040u);
                                if _e189_ {
                                    if ((_e187_ & 2147483647u) > 2139095040u) {
                                        phi_3201_ = f32();
                                        phi_3202_ = true;
                                    } else {
                                        let _e220_ = ((_e187_ & 2147483648u) == 0u);
                                        if _e220_ {
                                            let _e221_ = bitcast<u32>(_e186_);
                                            let _e222_ = (_e221_ & 2139095040u);
                                            if (_e222_ == 2139095040u) {
                                                phi_3193_ = true;
                                            } else {
                                                phi_3193_ = (_e222_ == 0u);
                                            }
                                            let _e226_ = phi_3193_;
                                            phi_3199_ = bitcast<f32>(select(_e221_, 0u, _e226_));
                                        } else {
                                            phi_3199_ = f32();
                                        }
                                        let _e230_ = phi_3199_;
                                        phi_3201_ = _e230_;
                                        phi_3202_ = select(true, false, _e220_);
                                    }
                                    let _e233_ = phi_3201_;
                                    let _e235_ = phi_3202_;
                                    if _e235_ {
                                        let _e236_ = bitcast<u32>(_e185_);
                                        let _e237_ = (_e236_ & 2139095040u);
                                        if (_e237_ == 2139095040u) {
                                            phi_3216_ = true;
                                        } else {
                                            phi_3216_ = (_e237_ == 0u);
                                        }
                                        let _e241_ = phi_3216_;
                                        phi_3222_ = bitcast<f32>(select(_e236_, 0u, _e241_));
                                    } else {
                                        phi_3222_ = _e233_;
                                    }
                                    let _e245_ = phi_3222_;
                                    phi_3281_ = _e245_;
                                } else {
                                    if (_e160_ < _e185_) {
                                        let _e207_ = bitcast<u32>(_e185_);
                                        let _e208_ = (_e207_ & 2139095040u);
                                        if (_e208_ == 2139095040u) {
                                            phi_3237_ = true;
                                        } else {
                                            phi_3237_ = (_e208_ == 0u);
                                        }
                                        let _e212_ = phi_3237_;
                                        phi_3280_ = bitcast<f32>(select(_e207_, 0u, _e212_));
                                    } else {
                                        if (_e160_ > _e186_) {
                                            let _e197_ = bitcast<u32>(_e186_);
                                            let _e198_ = (_e197_ & 2139095040u);
                                            if (_e198_ == 2139095040u) {
                                                phi_3257_ = true;
                                            } else {
                                                phi_3257_ = (_e198_ == 0u);
                                            }
                                            let _e202_ = phi_3257_;
                                            phi_3279_ = bitcast<f32>(select(_e197_, 0u, _e202_));
                                        } else {
                                            if _e189_ {
                                                phi_3273_ = true;
                                            } else {
                                                phi_3273_ = (_e188_ == 0u);
                                            }
                                            let _e194_ = phi_3273_;
                                            phi_3279_ = bitcast<f32>(select(_e187_, 0u, _e194_));
                                        }
                                        let _e206_ = phi_3279_;
                                        phi_3280_ = _e206_;
                                    }
                                    let _e216_ = phi_3280_;
                                    phi_3281_ = _e216_;
                                }
                                let _e247_ = phi_3281_;
                                phi_3300_ = _e247_;
                            } else {
                                let _e177_ = bitcast<u32>((0.5f * (_e170_ + _e171_)));
                                let _e178_ = (_e177_ & 2139095040u);
                                if (_e178_ == 2139095040u) {
                                    phi_3294_ = true;
                                } else {
                                    phi_3294_ = (_e178_ == 0u);
                                }
                                let _e182_ = phi_3294_;
                                phi_3300_ = bitcast<f32>(select(_e177_, 0u, _e182_));
                            }
                            let _e249_ = phi_3300_;
                            if (_e57_ == 0u) {
                                phi_3311_ = 0u;
                            } else {
                                let _e251_ = (_e57_ - 1u);
                                phi_3311_ = select(_e251_, _e103_, (_e103_ < _e251_));
                            }
                            let _e255_ = phi_3311_;
                            if (_e255_ < _e57_) {
                            } else {
                                phi_3858_ = true;
                                break;
                            }
                            theta.member[_e255_] = _e249_;
                            phi_3025_ = (_e103_ + 256u);
                        } else {
                            phi_3025_ = u32();
                        }
                        let _e261_ = phi_3025_;
                        phi_3024_ = _e261_;
                        phi_3858_ = false;
                        if !(_e104_) {
                            break;
                        }
                    }
                    let _e264_ = phi_3858_;
                    if _e264_ {
                        break;
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn inner_opt_step(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
