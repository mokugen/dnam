struct type_6_ {
    member: array<f32>,
}

struct type_12_ {
    member: u32,
    member_1_: u32,
}

struct type_14_ {
    member: f32,
    member_1_: f32,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: u32,
    member_6_: u32,
    member_7_: u32,
}

struct type_16_ {
    member: type_15_,
}

struct type_19_ {
    member: array<vec2<f32>>,
}

struct type_22_ {
    member: array<i32>,
}

struct type_28_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: bool,
    member_5_: bool,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> sg_lid_1_: u32;
@group(0) @binding(0) 
var<uniform> uniforms: type_16_;
@group(0) @binding(1) 
var<storage> features: type_6_;
@group(0) @binding(2) 
var<storage> targets: type_6_;
@group(0) @binding(3) 
var<storage> col_scales: type_19_;
@group(0) @binding(4) 
var<storage, read_write> ozaki_buckets: type_22_;
var<workgroup> reduce_i32_: array<i32, 1280>;

fn function_() {
    var local: array<i32, 5>;
    var local_1_: array<i32, 5>;
    var phi_242_: u32;
    var phi_245_: u32;
    var phi_263_: u32;
    var phi_243_: u32;
    var phi_246_: u32;
    var local_2_: u32;
    var local_3_: u32;
    var local_4_: u32;
    var phi_269_: type_28_;
    var phi_270_: type_28_;
    var phi_271_: bool;
    var phi_276_: type_28_;
    var phi_195_: u32;
    var phi_198_: u32;
    var phi_220_: u32;
    var phi_196_: u32;
    var phi_199_: u32;
    var local_5_: u32;
    var local_6_: u32;
    var local_7_: u32;
    var local_8_: u32;
    var phi_277_: type_28_;
    var phi_278_: type_28_;
    var phi_279_: bool;
    var phi_283_: type_28_;
    var phi_284_: type_28_;
    var phi_1269_: u32;
    var phi_1272_: type_12_;
    var phi_1274_: type_12_;
    var phi_1299_: u32;
    var phi_1318_: u32;
    var phi_1337_: u32;
    var phi_1356_: u32;
    var phi_1375_: u32;
    var phi_1514_: type_12_;
    var phi_1515_: type_12_;
    var phi_1270_: u32;
    var phi_1273_: type_12_;
    var phi_1275_: type_12_;
    var phi_1938_: bool;
    var local_9_: type_12_;
    var local_10_: type_12_;
    var local_11_: type_12_;
    var local_12_: type_12_;
    var phi_1554_: u32;
    var phi_1555_: u32;
    var phi_1972_: bool;
    var phi_1727_: u32;
    var phi_1747_: u32;
    var phi_1767_: u32;
    var phi_1787_: u32;
    var phi_1808_: type_12_;
    var phi_1809_: type_12_;
    var phi_1821_: u32;
    var phi_1841_: u32;
    var phi_1861_: u32;
    var phi_1881_: u32;
    var phi_303_: u32;
    var phi_325_: u32;
    var phi_367_: bool;
    var phi_368_: bool;
    var phi_369_: f32;
    var phi_371_: bool;
    var phi_372_: f32;
    var phi_373_: bool;
    var phi_374_: bool;
    var phi_381_: f32;
    var phi_1918_: bool;
    var phi_418_: u32;
    var phi_425_: type_14_;
    var phi_426_: type_14_;
    var phi_427_: bool;
    var phi_428_: type_14_;
    var phi_429_: bool;
    var phi_434_: type_14_;
    var phi_436_: f32;
    var phi_468_: bool;
    var phi_469_: bool;
    var phi_470_: f32;
    var phi_472_: bool;
    var phi_473_: f32;
    var phi_474_: bool;
    var phi_475_: bool;
    var phi_482_: f32;
    var phi_1933_: bool;
    var phi_519_: u32;
    var phi_526_: type_14_;
    var phi_527_: type_14_;
    var phi_528_: bool;
    var phi_529_: type_14_;
    var phi_530_: bool;
    var phi_535_: type_14_;
    var phi_537_: f32;
    var phi_544_: bool;
    var phi_547_: type_12_;
    var phi_550_: array<i32, 6>;
    var phi_548_: type_12_;
    var phi_573_: type_12_;
    var phi_608_: u32;
    var phi_654_: u32;
    var phi_633_: u32;
    var phi_662_: f32;
    var phi_737_: i32;
    var phi_767_: i32;
    var phi_797_: i32;
    var phi_800_: array<i32, 3>;
    var phi_801_: array<i32, 3>;
    var phi_803_: array<i32, 3>;
    var phi_804_: bool;
    var phi_809_: array<i32, 3>;
    var phi_884_: i32;
    var phi_914_: i32;
    var phi_944_: i32;
    var phi_947_: array<i32, 3>;
    var phi_948_: array<i32, 3>;
    var phi_950_: array<i32, 3>;
    var phi_951_: bool;
    var phi_956_: array<i32, 3>;
    var phi_1003_: array<i32, 6>;
    var phi_1004_: array<i32, 6>;
    var phi_1005_: array<i32, 6>;
    var phi_551_: array<i32, 6>;
    var phi_1976_: bool;
    var local_13_: array<i32, 6>;
    var local_14_: array<i32, 6>;
    var local_15_: array<i32, 6>;
    var local_16_: array<i32, 6>;
    var local_17_: array<i32, 6>;
    var phi_1027_: type_12_;
    var phi_1028_: type_12_;
    var phi_1051_: type_12_;
    var phi_2059_: bool;
    var phi_2070_: bool;
    var phi_1086_: u32;
    var phi_1101_: type_12_;
    var phi_1102_: type_12_;
    var phi_1125_: type_12_;
    var phi_2066_: bool;
    var phi_2073_: bool;
    var phi_2072_: bool;
    var phi_1087_: u32;
    var phi_2071_: bool;
    var phi_1190_: type_12_;
    var phi_1191_: type_12_;
    var phi_1214_: type_12_;
    var phi_1240_: u32;
    var phi_2074_: bool;
    var phi_1258_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e74_ = wg_id_1_;
            let _e75_ = lid_1_;
            let _e76_ = sg_lid_1_;
            let _e78_ = arrayLength((&features.member));
            let _e80_ = arrayLength((&targets.member));
            let _e82_ = arrayLength((&col_scales.member));
            let _e84_ = arrayLength((&ozaki_buckets.member));
            let _e85_ = (_e75_ & 255u);
            let _e88_ = uniforms.member.member_5_;
            let _e94_ = ((_e74_.y * select(256u, _e88_, (_e88_ > 0u))) + _e74_.x);
            let _e97_ = uniforms.member.member;
            let _e100_ = uniforms.member.member_1_;
            let _e103_ = uniforms.member.member_2_;
            let _e106_ = uniforms.member.member_3_;
            let _e108_ = select(_e106_, 1024u, (1024u < _e106_));
            let _e111_ = uniforms.member.member_4_;
            let _e113_ = select(1u, _e111_, (1u < _e111_));
            let _e117_ = ((_e100_ * (_e100_ + 1u)) >> bitcast<u32>(1i));
            let _e118_ = (_e100_ * _e103_);
            let _e120_ = ((_e117_ + _e118_) * 4u);
            if (_e100_ == 0u) {
                phi_284_ = type_28_(0u, 0u, 0u, 0u, false, true);
            } else {
                if (_e94_ < _e117_) {
                    phi_195_ = (_e100_ - 1u);
                    phi_198_ = 0u;
                    loop {
                        let _e168_ = phi_195_;
                        let _e170_ = phi_198_;
                        local_5_ = _e170_;
                        local_6_ = _e170_;
                        local_7_ = _e170_;
                        local_8_ = _e170_;
                        let _e171_ = (_e170_ < _e168_);
                        if _e171_ {
                            let _e176_ = (_e170_ + (((_e168_ - _e170_) + 1u) >> bitcast<u32>(1i)));
                            let _e177_ = (_e100_ - _e176_);
                            let _e183_ = ((_e117_ - ((_e177_ * (_e177_ + 1u)) >> bitcast<u32>(1i))) <= _e94_);
                            if _e183_ {
                                phi_220_ = _e168_;
                            } else {
                                phi_220_ = (_e176_ - 1u);
                            }
                            let _e186_ = phi_220_;
                            phi_196_ = _e186_;
                            phi_199_ = select(_e170_, _e176_, _e183_);
                        } else {
                            phi_196_ = u32();
                            phi_199_ = u32();
                        }
                        let _e189_ = phi_196_;
                        let _e191_ = phi_199_;
                        phi_195_ = _e189_;
                        phi_198_ = _e191_;
                        if !(_e171_) {
                            break;
                        }
                    }
                    let _e194_ = local_5_;
                    let _e195_ = (_e100_ - _e194_);
                    let _e203_ = local_6_;
                    let _e204_ = (_e203_ + (_e94_ - (_e117_ - ((_e195_ * (_e195_ + 1u)) >> bitcast<u32>(1i)))));
                    let _e206_ = local_7_;
                    let _e208_ = local_8_;
                    phi_277_ = type_28_(_e206_, _e204_, _e208_, _e204_, true, true);
                    phi_278_ = type_28_();
                    phi_279_ = true;
                } else {
                    let _e123_ = (_e94_ - _e117_);
                    if (_e103_ == 0u) {
                        phi_270_ = type_28_();
                        phi_271_ = true;
                    } else {
                        let _e125_ = (_e123_ >= _e118_);
                        if _e125_ {
                            phi_269_ = type_28_();
                        } else {
                            phi_242_ = (_e100_ - 1u);
                            phi_245_ = 0u;
                            loop {
                                let _e128_ = phi_242_;
                                let _e130_ = phi_245_;
                                local_2_ = _e130_;
                                local_3_ = _e130_;
                                local_4_ = _e130_;
                                let _e131_ = (_e130_ < _e128_);
                                if _e131_ {
                                    let _e136_ = (_e130_ + (((_e128_ - _e130_) + 1u) >> bitcast<u32>(1i)));
                                    let _e138_ = ((_e136_ * _e103_) <= _e123_);
                                    if _e138_ {
                                        phi_263_ = _e128_;
                                    } else {
                                        phi_263_ = (_e136_ - 1u);
                                    }
                                    let _e141_ = phi_263_;
                                    phi_243_ = _e141_;
                                    phi_246_ = select(_e130_, _e136_, _e138_);
                                } else {
                                    phi_243_ = u32();
                                    phi_246_ = u32();
                                }
                                let _e144_ = phi_243_;
                                let _e146_ = phi_246_;
                                phi_242_ = _e144_;
                                phi_245_ = _e146_;
                                if !(_e131_) {
                                    break;
                                }
                            }
                            let _e149_ = local_2_;
                            let _e151_ = (_e123_ - (_e149_ * _e103_));
                            let _e154_ = local_3_;
                            let _e156_ = local_4_;
                            phi_269_ = type_28_(_e154_, _e151_, _e156_, (_e100_ + _e151_), true, false);
                        }
                        let _e159_ = phi_269_;
                        phi_270_ = _e159_;
                        phi_271_ = _e125_;
                    }
                    let _e161_ = phi_270_;
                    let _e163_ = phi_271_;
                    if _e163_ {
                        phi_276_ = type_28_(0u, 0u, 0u, 0u, false, false);
                    } else {
                        phi_276_ = type_28_();
                    }
                    let _e165_ = phi_276_;
                    phi_277_ = _e165_;
                    phi_278_ = _e161_;
                    phi_279_ = _e163_;
                }
                let _e211_ = phi_277_;
                let _e213_ = phi_278_;
                let _e215_ = phi_279_;
                if _e215_ {
                    phi_283_ = _e211_;
                } else {
                    phi_283_ = _e213_;
                }
                let _e217_ = phi_283_;
                phi_284_ = _e217_;
            }
            let _e219_ = phi_284_;
            let _e222_ = uniforms.member.member_6_;
            if (_e222_ == 0u) {
                let _e580_ = (_e82_ == 0u);
                if _e580_ {
                    phi_303_ = 0u;
                } else {
                    let _e581_ = (_e82_ - 1u);
                    phi_303_ = select(_e581_, _e219_.member_2_, (_e219_.member_2_ < _e581_));
                }
                let _e585_ = phi_303_;
                if (_e585_ < _e82_) {
                } else {
                    break;
                }
                let _e590_ = col_scales.member[_e585_].x;
                let _e594_ = col_scales.member[_e585_].y;
                if _e580_ {
                    phi_325_ = 0u;
                } else {
                    let _e596_ = (_e82_ - 1u);
                    phi_325_ = select(_e596_, _e219_.member_3_, (_e219_.member_3_ < _e596_));
                }
                let _e600_ = phi_325_;
                if (_e600_ < _e82_) {
                } else {
                    break;
                }
                let _e605_ = col_scales.member[_e600_].x;
                let _e609_ = col_scales.member[_e600_].y;
                let _e610_ = bitcast<u32>(_e594_);
                let _e613_ = ((_e610_ >> bitcast<u32>(23u)) & 255u);
                if ((_e610_ & 2155872255u) == 0u) {
                    let _e616_ = (_e613_ > 0u);
                    if _e616_ {
                        let _e618_ = bitcast<u32>(_e590_);
                        let _e620_ = bitcast<f32>((_e618_ & 2147483647u));
                        if (_e613_ < 255u) {
                            if ((_e618_ & 2139095040u) != 2139095040u) {
                                phi_367_ = select(true, false, ((_e620_ * _e594_) <= 1f));
                            } else {
                                phi_367_ = true;
                            }
                            let _e627_ = phi_367_;
                            phi_368_ = _e627_;
                        } else {
                            phi_368_ = true;
                        }
                        let _e629_ = phi_368_;
                        phi_369_ = _e620_;
                        phi_371_ = _e629_;
                    } else {
                        phi_369_ = f32();
                        phi_371_ = false;
                    }
                    let _e631_ = phi_369_;
                    let _e633_ = phi_371_;
                    phi_372_ = _e631_;
                    phi_373_ = select(true, false, _e616_);
                    phi_374_ = _e633_;
                } else {
                    phi_372_ = f32();
                    phi_373_ = true;
                    phi_374_ = false;
                }
                let _e636_ = phi_372_;
                let _e638_ = phi_373_;
                let _e640_ = phi_374_;
                if _e638_ {
                    phi_381_ = bitcast<f32>((bitcast<u32>(_e590_) & 2147483647u));
                } else {
                    phi_381_ = _e636_;
                }
                let _e645_ = phi_381_;
                if select(_e640_, true, _e638_) {
                    let _e647_ = bitcast<u32>(_e645_);
                    if ((_e647_ & 2147483648u) == 0u) {
                        if (_e647_ == 0u) {
                            phi_426_ = type_14_();
                            phi_427_ = true;
                        } else {
                            let _e651_ = (_e647_ >= 2139095040u);
                            if _e651_ {
                                phi_425_ = type_14_();
                            } else {
                                if (_e645_ != _e645_) {
                                    phi_1918_ = true;
                                } else {
                                    phi_1918_ = (85070590000000000000000000000000000000f <= _e645_);
                                }
                                let _e655_ = phi_1918_;
                                let _e661_ = (((bitcast<u32>(select(_e645_, 85070590000000000000000000000000000000f, _e655_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                                if (_e661_ < 1u) {
                                    phi_418_ = 1u;
                                } else {
                                    phi_418_ = select(_e661_, 253u, (_e661_ > 253u));
                                }
                                let _e666_ = phi_418_;
                                phi_425_ = type_14_(bitcast<f32>((_e666_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e666_) << bitcast<u32>(23u))));
                            }
                            let _e676_ = phi_425_;
                            phi_426_ = _e676_;
                            phi_427_ = _e651_;
                        }
                        let _e678_ = phi_426_;
                        let _e680_ = phi_427_;
                        phi_428_ = _e678_;
                        phi_429_ = _e680_;
                    } else {
                        phi_428_ = type_14_();
                        phi_429_ = true;
                    }
                    let _e682_ = phi_428_;
                    let _e684_ = phi_429_;
                    if _e684_ {
                        phi_434_ = type_14_(1f, 1f);
                    } else {
                        phi_434_ = _e682_;
                    }
                    let _e686_ = phi_434_;
                    phi_436_ = _e686_.member_1_;
                } else {
                    phi_436_ = _e594_;
                }
                let _e689_ = phi_436_;
                let _e690_ = bitcast<u32>(_e609_);
                let _e693_ = ((_e690_ >> bitcast<u32>(23u)) & 255u);
                if ((_e690_ & 2155872255u) == 0u) {
                    let _e696_ = (_e693_ > 0u);
                    if _e696_ {
                        let _e698_ = bitcast<u32>(_e605_);
                        let _e700_ = bitcast<f32>((_e698_ & 2147483647u));
                        if (_e693_ < 255u) {
                            if ((_e698_ & 2139095040u) != 2139095040u) {
                                phi_468_ = select(true, false, ((_e700_ * _e609_) <= 1f));
                            } else {
                                phi_468_ = true;
                            }
                            let _e707_ = phi_468_;
                            phi_469_ = _e707_;
                        } else {
                            phi_469_ = true;
                        }
                        let _e709_ = phi_469_;
                        phi_470_ = _e700_;
                        phi_472_ = _e709_;
                    } else {
                        phi_470_ = f32();
                        phi_472_ = false;
                    }
                    let _e711_ = phi_470_;
                    let _e713_ = phi_472_;
                    phi_473_ = _e711_;
                    phi_474_ = select(true, false, _e696_);
                    phi_475_ = _e713_;
                } else {
                    phi_473_ = f32();
                    phi_474_ = true;
                    phi_475_ = false;
                }
                let _e716_ = phi_473_;
                let _e718_ = phi_474_;
                let _e720_ = phi_475_;
                if _e718_ {
                    phi_482_ = bitcast<f32>((bitcast<u32>(_e605_) & 2147483647u));
                } else {
                    phi_482_ = _e716_;
                }
                let _e725_ = phi_482_;
                if select(_e720_, true, _e718_) {
                    let _e727_ = bitcast<u32>(_e725_);
                    if ((_e727_ & 2147483648u) == 0u) {
                        if (_e727_ == 0u) {
                            phi_527_ = type_14_();
                            phi_528_ = true;
                        } else {
                            let _e731_ = (_e727_ >= 2139095040u);
                            if _e731_ {
                                phi_526_ = type_14_();
                            } else {
                                if (_e725_ != _e725_) {
                                    phi_1933_ = true;
                                } else {
                                    phi_1933_ = (85070590000000000000000000000000000000f <= _e725_);
                                }
                                let _e735_ = phi_1933_;
                                let _e741_ = (((bitcast<u32>(select(_e725_, 85070590000000000000000000000000000000f, _e735_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                                if (_e741_ < 1u) {
                                    phi_519_ = 1u;
                                } else {
                                    phi_519_ = select(_e741_, 253u, (_e741_ > 253u));
                                }
                                let _e746_ = phi_519_;
                                phi_526_ = type_14_(bitcast<f32>((_e746_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e746_) << bitcast<u32>(23u))));
                            }
                            let _e756_ = phi_526_;
                            phi_527_ = _e756_;
                            phi_528_ = _e731_;
                        }
                        let _e758_ = phi_527_;
                        let _e760_ = phi_528_;
                        phi_529_ = _e758_;
                        phi_530_ = _e760_;
                    } else {
                        phi_529_ = type_14_();
                        phi_530_ = true;
                    }
                    let _e762_ = phi_529_;
                    let _e764_ = phi_530_;
                    if _e764_ {
                        phi_535_ = type_14_(1f, 1f);
                    } else {
                        phi_535_ = _e762_;
                    }
                    let _e766_ = phi_535_;
                    phi_537_ = _e766_.member_1_;
                } else {
                    phi_537_ = _e609_;
                }
                let _e769_ = phi_537_;
                if _e219_.member_4_ {
                    phi_544_ = (_e74_.z < _e113_);
                } else {
                    phi_544_ = false;
                }
                let _e774_ = phi_544_;
                phi_547_ = type_12_(0u, 4u);
                phi_550_ = array<i32, 6>(0i, 0i, 0i, 0i, 0i, 0i);
                loop {
                    let _e776_ = phi_547_;
                    let _e778_ = phi_550_;
                    local_13_ = _e778_;
                    local_14_ = _e778_;
                    local_15_ = _e778_;
                    local_16_ = _e778_;
                    local_17_ = _e778_;
                    if (_e776_.member < _e776_.member_1_) {
                        phi_548_ = type_12_((_e776_.member + 1u), _e776_.member_1_);
                        phi_573_ = type_12_(1u, _e776_.member);
                    } else {
                        phi_548_ = _e776_;
                        phi_573_ = type_12_(0u, u32());
                    }
                    let _e791_ = phi_548_;
                    let _e793_ = phi_573_;
                    let _e797_ = (bitcast<i32>(_e793_.member) != 0i);
                    if _e797_ {
                        let _e799_ = ((_e793_.member_1_ * 256u) + _e85_);
                        let _e800_ = ((_e74_.z * _e108_) + _e799_);
                        if _e774_ {
                            if (_e799_ < _e108_) {
                                if (_e800_ < _e97_) {
                                    let _e805_ = ((_e219_.member * _e97_) + _e800_);
                                    let _e806_ = (_e78_ == 0u);
                                    if _e806_ {
                                        phi_608_ = 0u;
                                    } else {
                                        let _e807_ = (_e78_ - 1u);
                                        phi_608_ = select(_e807_, _e805_, (_e805_ < _e807_));
                                    }
                                    let _e811_ = phi_608_;
                                    if (_e811_ < _e78_) {
                                    } else {
                                        phi_1976_ = true;
                                        break;
                                    }
                                    let _e815_ = features.member[_e811_];
                                    if _e219_.member_5_ {
                                        let _e832_ = ((_e219_.member_1_ * _e97_) + _e800_);
                                        if _e806_ {
                                            phi_633_ = 0u;
                                        } else {
                                            let _e833_ = (_e78_ - 1u);
                                            phi_633_ = select(_e833_, _e832_, (_e832_ < _e833_));
                                        }
                                        let _e837_ = phi_633_;
                                        if (_e837_ < _e78_) {
                                        } else {
                                            phi_1976_ = true;
                                            break;
                                        }
                                        let _e841_ = features.member[_e837_];
                                        phi_662_ = _e841_;
                                    } else {
                                        let _e819_ = ((_e219_.member_1_ * _e97_) + _e800_);
                                        if (_e80_ == 0u) {
                                            phi_654_ = 0u;
                                        } else {
                                            let _e821_ = (_e80_ - 1u);
                                            phi_654_ = select(_e821_, _e819_, (_e819_ < _e821_));
                                        }
                                        let _e825_ = phi_654_;
                                        if (_e825_ < _e80_) {
                                        } else {
                                            phi_1976_ = true;
                                            break;
                                        }
                                        let _e829_ = targets.member[_e825_];
                                        phi_662_ = _e829_;
                                    }
                                    let _e843_ = phi_662_;
                                    if ((bitcast<u32>(_e815_) & 2139095040u) != 2139095040u) {
                                        let _e849_ = ((bitcast<u32>(_e689_) & 2139095040u) != 2139095040u);
                                        if _e849_ {
                                            let _e851_ = select(_e815_, -85070590000000000000000000000000000000f, (_e815_ < -85070590000000000000000000000000000000f));
                                            let _e854_ = (select(_e851_, 85070590000000000000000000000000000000f, (_e851_ > 85070590000000000000000000000000000000f)) * _e689_);
                                            if ((bitcast<u32>(_e854_) & 2139095040u) != 2139095040u) {
                                                let _e859_ = select(_e854_, -1f, (_e854_ < -1f));
                                                let _e861_ = select(_e859_, 1f, (_e859_ > 1f));
                                                let _e862_ = (_e861_ * 256f);
                                                if ((bitcast<u32>(_e862_) & 2139095040u) != 2139095040u) {
                                                    let _e867_ = select(_e862_, -4194304f, (_e862_ < -4194304f));
                                                    phi_737_ = (bitcast<i32>(bitcast<u32>((select(_e867_, 4194304f, (_e867_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_737_ = 0i;
                                                }
                                                let _e875_ = phi_737_;
                                                let _e878_ = (_e861_ - (f32(_e875_) * 0.00390625f));
                                                let _e879_ = (_e878_ * 65536f);
                                                if ((bitcast<u32>(_e879_) & 2139095040u) != 2139095040u) {
                                                    let _e884_ = select(_e879_, -4194304f, (_e879_ < -4194304f));
                                                    phi_767_ = (bitcast<i32>(bitcast<u32>((select(_e884_, 4194304f, (_e884_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_767_ = 0i;
                                                }
                                                let _e892_ = phi_767_;
                                                let _e896_ = ((_e878_ - (f32(_e892_) * 0.000015258789f)) * 16777216f);
                                                if ((bitcast<u32>(_e896_) & 2139095040u) != 2139095040u) {
                                                    let _e901_ = select(_e896_, -4194304f, (_e896_ < -4194304f));
                                                    phi_797_ = (bitcast<i32>(bitcast<u32>((select(_e901_, 4194304f, (_e901_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_797_ = 0i;
                                                }
                                                let _e909_ = phi_797_;
                                                phi_800_ = array<i32, 3>(_e875_, _e892_, _e909_);
                                            } else {
                                                phi_800_ = array<i32, 3>(0i, 0i, 0i);
                                            }
                                            let _e912_ = phi_800_;
                                            phi_801_ = _e912_;
                                        } else {
                                            phi_801_ = array<i32, 3>();
                                        }
                                        let _e914_ = phi_801_;
                                        phi_803_ = _e914_;
                                        phi_804_ = select(true, false, _e849_);
                                    } else {
                                        phi_803_ = array<i32, 3>();
                                        phi_804_ = true;
                                    }
                                    let _e917_ = phi_803_;
                                    let _e919_ = phi_804_;
                                    if _e919_ {
                                        phi_809_ = array<i32, 3>(0i, 0i, 0i);
                                    } else {
                                        phi_809_ = _e917_;
                                    }
                                    let _e921_ = phi_809_;
                                    if ((bitcast<u32>(_e843_) & 2139095040u) != 2139095040u) {
                                        let _e927_ = ((bitcast<u32>(_e769_) & 2139095040u) != 2139095040u);
                                        if _e927_ {
                                            let _e929_ = select(_e843_, -85070590000000000000000000000000000000f, (_e843_ < -85070590000000000000000000000000000000f));
                                            let _e932_ = (select(_e929_, 85070590000000000000000000000000000000f, (_e929_ > 85070590000000000000000000000000000000f)) * _e769_);
                                            if ((bitcast<u32>(_e932_) & 2139095040u) != 2139095040u) {
                                                let _e937_ = select(_e932_, -1f, (_e932_ < -1f));
                                                let _e939_ = select(_e937_, 1f, (_e937_ > 1f));
                                                let _e940_ = (_e939_ * 256f);
                                                if ((bitcast<u32>(_e940_) & 2139095040u) != 2139095040u) {
                                                    let _e945_ = select(_e940_, -4194304f, (_e940_ < -4194304f));
                                                    phi_884_ = (bitcast<i32>(bitcast<u32>((select(_e945_, 4194304f, (_e945_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_884_ = 0i;
                                                }
                                                let _e953_ = phi_884_;
                                                let _e956_ = (_e939_ - (f32(_e953_) * 0.00390625f));
                                                let _e957_ = (_e956_ * 65536f);
                                                if ((bitcast<u32>(_e957_) & 2139095040u) != 2139095040u) {
                                                    let _e962_ = select(_e957_, -4194304f, (_e957_ < -4194304f));
                                                    phi_914_ = (bitcast<i32>(bitcast<u32>((select(_e962_, 4194304f, (_e962_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_914_ = 0i;
                                                }
                                                let _e970_ = phi_914_;
                                                let _e974_ = ((_e956_ - (f32(_e970_) * 0.000015258789f)) * 16777216f);
                                                if ((bitcast<u32>(_e974_) & 2139095040u) != 2139095040u) {
                                                    let _e979_ = select(_e974_, -4194304f, (_e974_ < -4194304f));
                                                    phi_944_ = (bitcast<i32>(bitcast<u32>((select(_e979_, 4194304f, (_e979_ > 4194304f)) + 12582912f))) - 1262485504i);
                                                } else {
                                                    phi_944_ = 0i;
                                                }
                                                let _e987_ = phi_944_;
                                                phi_947_ = array<i32, 3>(_e953_, _e970_, _e987_);
                                            } else {
                                                phi_947_ = array<i32, 3>(0i, 0i, 0i);
                                            }
                                            let _e990_ = phi_947_;
                                            phi_948_ = _e990_;
                                        } else {
                                            phi_948_ = array<i32, 3>();
                                        }
                                        let _e992_ = phi_948_;
                                        phi_950_ = _e992_;
                                        phi_951_ = select(true, false, _e927_);
                                    } else {
                                        phi_950_ = array<i32, 3>();
                                        phi_951_ = true;
                                    }
                                    let _e995_ = phi_950_;
                                    let _e997_ = phi_951_;
                                    if _e997_ {
                                        phi_956_ = array<i32, 3>(0i, 0i, 0i);
                                    } else {
                                        phi_956_ = _e995_;
                                    }
                                    let _e999_ = phi_956_;
                                    let _e1011_ = array<i32, 6>((_e778_[0] + (_e921_[0] * _e999_[0])), _e778_[1], _e778_[2], _e778_[3], _e778_[4], _e778_[5]);
                                    let _e1025_ = array<i32, 6>(_e1011_[0], (_e778_[1] + ((_e921_[0] * _e999_[1]) + (_e921_[1] * _e999_[0]))), _e1011_[2], _e1011_[3], _e1011_[4], _e1011_[5]);
                                    let _e1041_ = array<i32, 6>(_e1025_[0], _e1025_[1], (_e778_[2] + (((_e921_[0] * _e999_[2]) + (_e921_[1] * _e999_[1])) + (_e921_[2] * _e999_[0]))), _e1025_[3], _e1025_[4], _e1025_[5]);
                                    let _e1053_ = array<i32, 6>(_e1041_[0], _e1041_[1], _e1041_[2], (_e778_[3] + ((_e921_[1] * _e999_[2]) + (_e921_[2] * _e999_[1]))), _e1041_[4], _e1041_[5]);
                                    phi_1003_ = array<i32, 6>(_e1053_[0], _e1053_[1], _e1053_[2], _e1053_[3], (_e778_[4] + (_e921_[2] * _e999_[2])), _e1053_[5]);
                                } else {
                                    phi_1003_ = _e778_;
                                }
                                let _e1065_ = phi_1003_;
                                phi_1004_ = _e1065_;
                            } else {
                                phi_1004_ = _e778_;
                            }
                            let _e1067_ = phi_1004_;
                            phi_1005_ = _e1067_;
                        } else {
                            phi_1005_ = _e778_;
                        }
                        let _e1069_ = phi_1005_;
                        phi_551_ = _e1069_;
                    } else {
                        phi_551_ = array<i32, 6>();
                    }
                    let _e1071_ = phi_551_;
                    phi_547_ = _e791_;
                    phi_550_ = _e1071_;
                    phi_1976_ = false;
                    if !(_e797_) {
                        break;
                    }
                }
                let _e1074_ = phi_1976_;
                if _e1074_ {
                    break;
                }
                let _e1076_ = local_13_;
                let _e1079_ = local_14_;
                let _e1082_ = local_15_;
                let _e1085_ = local_16_;
                let _e1088_ = local_17_;
                let _e1047 = subgroupAdd(_e1076_[0]);
                let _e1049 = subgroupAdd(_e1079_[1]);
                let _e1051 = subgroupAdd(_e1082_[2]);
                let _e1053 = subgroupAdd(_e1085_[3]);
                let _e1055 = subgroupAdd(_e1088_[4]);
                if (_e76_ == 0u) {
                    local[0] = _e1047;
                    local[1] = _e1049;
                    local[2] = _e1051;
                    local[3] = _e1053;
                    local[4] = _e1055;
                } else {
                    local = array<i32, 5>(0i, 0i, 0i, 0i, 0i);
                }
                phi_1027_ = type_12_(0u, 5u);
                loop {
                    let _e1102_ = phi_1027_;
                    if (_e1102_.member < _e1102_.member_1_) {
                        phi_1028_ = type_12_((_e1102_.member + 1u), _e1102_.member_1_);
                        phi_1051_ = type_12_(1u, _e1102_.member);
                    } else {
                        phi_1028_ = _e1102_;
                        phi_1051_ = type_12_(0u, u32());
                    }
                    let _e1115_ = phi_1028_;
                    let _e1117_ = phi_1051_;
                    let _e1121_ = (bitcast<i32>(_e1117_.member) != 0i);
                    if _e1121_ {
                        if (_e1117_.member_1_ < 5u) {
                        } else {
                            phi_2059_ = true;
                            break;
                        }
                        let _e1124_ = local[_e1117_.member_1_];
                        let _e1126_ = ((_e1117_.member_1_ * 256u) + _e85_);
                        let _e1128_ = select(1279u, _e1126_, (_e1126_ < 1279u));
                        if (_e1128_ < 1280u) {
                        } else {
                            phi_2059_ = true;
                            break;
                        }
                        reduce_i32_[_e1128_] = _e1124_;
                    }
                    phi_1027_ = _e1115_;
                    phi_2059_ = _e1074_;
                    if !(_e1121_) {
                        break;
                    }
                }
                let _e1133_ = phi_2059_;
                if _e1133_ {
                    break;
                }
                workgroupBarrier();
                phi_2070_ = _e1133_;
                phi_1086_ = 128u;
                loop {
                    let _e1135_ = phi_2070_;
                    let _e1137_ = phi_1086_;
                    let _e1138_ = (_e1137_ > 0u);
                    if _e1138_ {
                        if (_e85_ < _e1137_) {
                            phi_1101_ = type_12_(0u, 5u);
                            loop {
                                let _e1141_ = phi_1101_;
                                if (_e1141_.member < _e1141_.member_1_) {
                                    phi_1102_ = type_12_((_e1141_.member + 1u), _e1141_.member_1_);
                                    phi_1125_ = type_12_(1u, _e1141_.member);
                                } else {
                                    phi_1102_ = _e1141_;
                                    phi_1125_ = type_12_(0u, u32());
                                }
                                let _e1154_ = phi_1102_;
                                let _e1156_ = phi_1125_;
                                let _e1160_ = (bitcast<i32>(_e1156_.member) != 0i);
                                if _e1160_ {
                                    let _e1162_ = ((_e1156_.member_1_ * 256u) + _e85_);
                                    let _e1163_ = (_e1162_ < 1280u);
                                    if _e1163_ {
                                    } else {
                                        phi_2066_ = true;
                                        break;
                                    }
                                    let _e1165_ = reduce_i32_[_e1162_];
                                    let _e1166_ = (_e1162_ + _e1137_);
                                    if (_e1166_ < 1280u) {
                                    } else {
                                        phi_2066_ = true;
                                        break;
                                    }
                                    let _e1169_ = reduce_i32_[_e1166_];
                                    if _e1163_ {
                                    } else {
                                        phi_2066_ = true;
                                        break;
                                    }
                                    reduce_i32_[_e1162_] = (_e1165_ + _e1169_);
                                }
                                phi_1101_ = _e1154_;
                                phi_2066_ = _e1135_;
                                if !(_e1160_) {
                                    break;
                                }
                            }
                            let _e1173_ = phi_2066_;
                            phi_2071_ = _e1173_;
                            if _e1173_ {
                                break;
                            }
                            phi_2073_ = _e1173_;
                        } else {
                            phi_2073_ = _e1135_;
                        }
                        let _e1175_ = phi_2073_;
                        workgroupBarrier();
                        phi_2072_ = _e1175_;
                        phi_1087_ = (_e1137_ >> bitcast<u32>(1i));
                    } else {
                        phi_2072_ = _e1135_;
                        phi_1087_ = u32();
                    }
                    let _e1179_ = phi_2072_;
                    let _e1181_ = phi_1087_;
                    phi_2070_ = _e1179_;
                    phi_1086_ = _e1181_;
                    phi_2071_ = _e1179_;
                    if !(_e1138_) {
                        break;
                    }
                }
                let _e1184_ = phi_2071_;
                if _e1184_ {
                    break;
                }
                let _e1186_ = reduce_i32_[0];
                let _e1188_ = reduce_i32_[256];
                let _e1190_ = reduce_i32_[512];
                let _e1192_ = reduce_i32_[768];
                let _e1194_ = reduce_i32_[1024];
                workgroupBarrier();
                local_1_ = array<i32, 5>(_e1186_, _e1188_, _e1190_, _e1192_, _e1194_);
                if (_e85_ == 0u) {
                    if _e774_ {
                        let _e1200_ = (_e120_ + (((_e94_ * _e113_) + _e74_.z) * 6u));
                        phi_1190_ = type_12_(0u, 5u);
                        loop {
                            let _e1202_ = phi_1190_;
                            if (_e1202_.member < _e1202_.member_1_) {
                                phi_1191_ = type_12_((_e1202_.member + 1u), _e1202_.member_1_);
                                phi_1214_ = type_12_(1u, _e1202_.member);
                            } else {
                                phi_1191_ = _e1202_;
                                phi_1214_ = type_12_(0u, u32());
                            }
                            let _e1215_ = phi_1191_;
                            let _e1217_ = phi_1214_;
                            let _e1221_ = (bitcast<i32>(_e1217_.member) != 0i);
                            if _e1221_ {
                                if (_e1217_.member_1_ < 5u) {
                                } else {
                                    phi_2074_ = true;
                                    break;
                                }
                                let _e1224_ = local_1_[_e1217_.member_1_];
                                let _e1225_ = (_e1200_ + _e1217_.member_1_);
                                if (_e84_ == 0u) {
                                    phi_1240_ = 0u;
                                } else {
                                    let _e1227_ = (_e84_ - 1u);
                                    phi_1240_ = select(_e1227_, _e1225_, (_e1225_ < _e1227_));
                                }
                                let _e1231_ = phi_1240_;
                                if (_e1231_ < _e84_) {
                                } else {
                                    phi_2074_ = true;
                                    break;
                                }
                                ozaki_buckets.member[_e1231_] = _e1224_;
                            }
                            phi_1190_ = _e1215_;
                            phi_2074_ = _e1184_;
                            if !(_e1221_) {
                                break;
                            }
                        }
                        let _e1237_ = phi_2074_;
                        if _e1237_ {
                            break;
                        }
                        let _e1238_ = (_e1200_ + 5u);
                        if (_e84_ == 0u) {
                            phi_1258_ = 0u;
                        } else {
                            let _e1240_ = (_e84_ - 1u);
                            phi_1258_ = select(_e1240_, _e1238_, (_e1238_ < _e1240_));
                        }
                        let _e1244_ = phi_1258_;
                        if (_e1244_ < _e84_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e1244_] = 0i;
                    }
                }
            } else {
                phi_1269_ = _e85_;
                phi_1272_ = type_12_(0u, 0u);
                phi_1274_ = type_12_(0u, 0u);
                loop {
                    let _e226_ = phi_1269_;
                    let _e228_ = phi_1272_;
                    let _e230_ = phi_1274_;
                    local_9_ = _e230_;
                    local_10_ = _e230_;
                    local_11_ = _e228_;
                    local_12_ = _e228_;
                    let _e231_ = (_e226_ < _e113_);
                    if _e231_ {
                        if _e219_.member_4_ {
                            let _e235_ = (_e120_ + (((_e94_ * _e113_) + _e226_) * 6u));
                            let _e236_ = (_e84_ == 0u);
                            if _e236_ {
                                phi_1299_ = 0u;
                            } else {
                                let _e237_ = (_e84_ - 1u);
                                phi_1299_ = select(_e237_, _e235_, (_e235_ < _e237_));
                            }
                            let _e241_ = phi_1299_;
                            if (_e241_ < _e84_) {
                            } else {
                                phi_1938_ = true;
                                break;
                            }
                            let _e245_ = ozaki_buckets.member[_e241_];
                            let _e246_ = (_e235_ + 1u);
                            if _e236_ {
                                phi_1318_ = 0u;
                            } else {
                                let _e247_ = (_e84_ - 1u);
                                phi_1318_ = select(_e247_, _e246_, (_e246_ < _e247_));
                            }
                            let _e251_ = phi_1318_;
                            if (_e251_ < _e84_) {
                            } else {
                                phi_1938_ = true;
                                break;
                            }
                            let _e255_ = ozaki_buckets.member[_e251_];
                            let _e256_ = (_e235_ + 2u);
                            if _e236_ {
                                phi_1337_ = 0u;
                            } else {
                                let _e257_ = (_e84_ - 1u);
                                phi_1337_ = select(_e257_, _e256_, (_e256_ < _e257_));
                            }
                            let _e261_ = phi_1337_;
                            if (_e261_ < _e84_) {
                            } else {
                                phi_1938_ = true;
                                break;
                            }
                            let _e265_ = ozaki_buckets.member[_e261_];
                            let _e266_ = (_e235_ + 3u);
                            if _e236_ {
                                phi_1356_ = 0u;
                            } else {
                                let _e267_ = (_e84_ - 1u);
                                phi_1356_ = select(_e267_, _e266_, (_e266_ < _e267_));
                            }
                            let _e271_ = phi_1356_;
                            if (_e271_ < _e84_) {
                            } else {
                                phi_1938_ = true;
                                break;
                            }
                            let _e275_ = ozaki_buckets.member[_e271_];
                            let _e276_ = (_e235_ + 4u);
                            if _e236_ {
                                phi_1375_ = 0u;
                            } else {
                                let _e277_ = (_e84_ - 1u);
                                phi_1375_ = select(_e277_, _e276_, (_e276_ < _e277_));
                            }
                            let _e281_ = phi_1375_;
                            if (_e281_ < _e84_) {
                            } else {
                                phi_1938_ = true;
                                break;
                            }
                            let _e285_ = ozaki_buckets.member[_e281_];
                            let _e286_ = bitcast<u32>(_e245_);
                            let _e299_ = (_e230_.member + (_e286_ << bitcast<u32>(16u)));
                            let _e304_ = bitcast<u32>(_e255_);
                            let _e315_ = (_e299_ + (_e304_ << bitcast<u32>(8u)));
                            let _e324_ = (_e315_ + bitcast<u32>(_e265_));
                            let _e330_ = bitcast<u32>(_e275_);
                            let _e343_ = (_e228_.member + (_e330_ << bitcast<u32>(8u)));
                            let _e352_ = (_e343_ + bitcast<u32>(_e285_));
                            phi_1514_ = type_12_(_e352_, ((((_e228_.member_1_ + ((bitcast<u32>((_e275_ >> bitcast<u32>(31i))) << bitcast<u32>(8u)) | (_e330_ >> bitcast<u32>(24u)))) + select(0u, 1u, (_e343_ < _e228_.member))) + bitcast<u32>((_e285_ >> bitcast<u32>(31i)))) + select(0u, 1u, (_e352_ < _e343_))));
                            phi_1515_ = type_12_(_e324_, ((((((_e230_.member_1_ + ((bitcast<u32>((_e245_ >> bitcast<u32>(31i))) << bitcast<u32>(16u)) | (_e286_ >> bitcast<u32>(16u)))) + select(0u, 1u, (_e299_ < _e230_.member))) + ((bitcast<u32>((_e255_ >> bitcast<u32>(31i))) << bitcast<u32>(8u)) | (_e304_ >> bitcast<u32>(24u)))) + select(0u, 1u, (_e315_ < _e299_))) + bitcast<u32>((_e265_ >> bitcast<u32>(31i)))) + select(0u, 1u, (_e324_ < _e315_))));
                        } else {
                            phi_1514_ = _e228_;
                            phi_1515_ = _e230_;
                        }
                        let _e359_ = phi_1514_;
                        let _e361_ = phi_1515_;
                        phi_1270_ = (_e226_ + 256u);
                        phi_1273_ = _e359_;
                        phi_1275_ = _e361_;
                    } else {
                        phi_1270_ = u32();
                        phi_1273_ = type_12_();
                        phi_1275_ = type_12_();
                    }
                    let _e364_ = phi_1270_;
                    let _e366_ = phi_1273_;
                    let _e368_ = phi_1275_;
                    phi_1269_ = _e364_;
                    phi_1272_ = _e366_;
                    phi_1274_ = _e368_;
                    phi_1938_ = false;
                    if !(_e231_) {
                        break;
                    }
                }
                let _e371_ = phi_1938_;
                if _e371_ {
                    break;
                }
                let _e373_ = local_9_;
                let _e376_ = local_10_;
                let _e379_ = local_11_;
                let _e382_ = local_12_;
                let _e384_ = (_e85_ & 255u);
                let _e385_ = (_e384_ < 1280u);
                if _e385_ {
                } else {
                    break;
                }
                reduce_i32_[_e384_] = bitcast<i32>(_e373_.member);
                let _e388_ = (256u + _e384_);
                let _e389_ = (_e388_ < 1280u);
                if _e389_ {
                } else {
                    break;
                }
                reduce_i32_[_e388_] = bitcast<i32>(_e376_.member_1_);
                let _e392_ = (512u + _e384_);
                let _e393_ = (_e392_ < 1280u);
                if _e393_ {
                } else {
                    break;
                }
                reduce_i32_[_e392_] = bitcast<i32>(_e379_.member);
                let _e396_ = (768u + _e384_);
                let _e397_ = (_e396_ < 1280u);
                if _e397_ {
                } else {
                    break;
                }
                reduce_i32_[_e396_] = bitcast<i32>(_e382_.member_1_);
                workgroupBarrier();
                phi_1554_ = 128u;
                loop {
                    let _e401_ = phi_1554_;
                    let _e402_ = (_e401_ > 0u);
                    if _e402_ {
                        if (_e384_ < _e401_) {
                            let _e404_ = (_e384_ + _e401_);
                            if _e385_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e405_ = reduce_i32_[_e384_];
                            let _e406_ = bitcast<u32>(_e405_);
                            if _e389_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e407_ = reduce_i32_[_e388_];
                            if (_e404_ < 1280u) {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e411_ = reduce_i32_[_e404_];
                            let _e413_ = (256u + _e404_);
                            if (_e413_ < 1280u) {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e416_ = reduce_i32_[_e413_];
                            let _e418_ = (_e406_ + bitcast<u32>(_e411_));
                            if _e385_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            reduce_i32_[_e384_] = bitcast<i32>(_e418_);
                            if _e389_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            reduce_i32_[_e388_] = bitcast<i32>(((bitcast<u32>(_e407_) + bitcast<u32>(_e416_)) + select(0u, 1u, (_e418_ < _e406_))));
                            if _e393_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e425_ = reduce_i32_[_e392_];
                            let _e426_ = bitcast<u32>(_e425_);
                            if _e397_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e427_ = reduce_i32_[_e396_];
                            let _e429_ = (512u + _e404_);
                            if (_e429_ < 1280u) {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e432_ = reduce_i32_[_e429_];
                            let _e434_ = (768u + _e404_);
                            if (_e434_ < 1280u) {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            let _e437_ = reduce_i32_[_e434_];
                            let _e439_ = (_e426_ + bitcast<u32>(_e432_));
                            if _e393_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            reduce_i32_[_e392_] = bitcast<i32>(_e439_);
                            if _e397_ {
                            } else {
                                phi_1972_ = true;
                                break;
                            }
                            reduce_i32_[_e396_] = bitcast<i32>(((bitcast<u32>(_e427_) + bitcast<u32>(_e437_)) + select(0u, 1u, (_e439_ < _e426_))));
                        }
                        workgroupBarrier();
                        phi_1555_ = (_e401_ >> bitcast<u32>(1i));
                    } else {
                        phi_1555_ = u32();
                    }
                    let _e449_ = phi_1555_;
                    phi_1554_ = _e449_;
                    phi_1972_ = _e371_;
                    if !(_e402_) {
                        break;
                    }
                }
                let _e452_ = phi_1972_;
                if _e452_ {
                    break;
                }
                let _e454_ = reduce_i32_[0];
                let _e455_ = bitcast<u32>(_e454_);
                let _e457_ = reduce_i32_[256];
                let _e458_ = bitcast<u32>(_e457_);
                let _e460_ = reduce_i32_[512];
                let _e461_ = bitcast<u32>(_e460_);
                let _e463_ = reduce_i32_[768];
                let _e464_ = bitcast<u32>(_e463_);
                workgroupBarrier();
                if (_e85_ == 0u) {
                    if _e219_.member_4_ {
                        let _e468_ = (_e94_ * 4u);
                        let _e471_ = uniforms.member.member_7_;
                        if ((_e471_ & 1u) == 0u) {
                            phi_1808_ = type_12_(_e461_, _e464_);
                            phi_1809_ = type_12_(_e455_, _e458_);
                        } else {
                            let _e474_ = (_e84_ == 0u);
                            if _e474_ {
                                phi_1727_ = 0u;
                            } else {
                                let _e475_ = (_e84_ - 1u);
                                phi_1727_ = select(_e475_, _e468_, (_e468_ < _e475_));
                            }
                            let _e479_ = phi_1727_;
                            if (_e479_ < _e84_) {
                            } else {
                                break;
                            }
                            let _e483_ = ozaki_buckets.member[_e479_];
                            let _e485_ = (_e468_ + 1u);
                            if _e474_ {
                                phi_1747_ = 0u;
                            } else {
                                let _e486_ = (_e84_ - 1u);
                                phi_1747_ = select(_e486_, _e485_, (_e485_ < _e486_));
                            }
                            let _e490_ = phi_1747_;
                            if (_e490_ < _e84_) {
                            } else {
                                break;
                            }
                            let _e494_ = ozaki_buckets.member[_e490_];
                            let _e496_ = (_e468_ + 2u);
                            if _e474_ {
                                phi_1767_ = 0u;
                            } else {
                                let _e497_ = (_e84_ - 1u);
                                phi_1767_ = select(_e497_, _e496_, (_e496_ < _e497_));
                            }
                            let _e501_ = phi_1767_;
                            if (_e501_ < _e84_) {
                            } else {
                                break;
                            }
                            let _e505_ = ozaki_buckets.member[_e501_];
                            let _e507_ = (_e468_ + 3u);
                            if _e474_ {
                                phi_1787_ = 0u;
                            } else {
                                let _e508_ = (_e84_ - 1u);
                                phi_1787_ = select(_e508_, _e507_, (_e507_ < _e508_));
                            }
                            let _e512_ = phi_1787_;
                            if (_e512_ < _e84_) {
                            } else {
                                break;
                            }
                            let _e516_ = ozaki_buckets.member[_e512_];
                            let _e518_ = (_e455_ + bitcast<u32>(_e483_));
                            let _e524_ = (_e461_ + bitcast<u32>(_e505_));
                            phi_1808_ = type_12_(_e524_, ((_e464_ + bitcast<u32>(_e516_)) + select(0u, 1u, (_e524_ < _e461_))));
                            phi_1809_ = type_12_(_e518_, ((_e458_ + bitcast<u32>(_e494_)) + select(0u, 1u, (_e518_ < _e455_))));
                        }
                        let _e531_ = phi_1808_;
                        let _e533_ = phi_1809_;
                        let _e535_ = (_e84_ == 0u);
                        if _e535_ {
                            phi_1821_ = 0u;
                        } else {
                            let _e536_ = (_e84_ - 1u);
                            phi_1821_ = select(_e536_, _e468_, (_e468_ < _e536_));
                        }
                        let _e540_ = phi_1821_;
                        if (_e540_ < _e84_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e540_] = bitcast<i32>(_e533_.member);
                        let _e546_ = (_e468_ + 1u);
                        if _e535_ {
                            phi_1841_ = 0u;
                        } else {
                            let _e547_ = (_e84_ - 1u);
                            phi_1841_ = select(_e547_, _e546_, (_e546_ < _e547_));
                        }
                        let _e551_ = phi_1841_;
                        if (_e551_ < _e84_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e551_] = bitcast<i32>(_e533_.member_1_);
                        let _e557_ = (_e468_ + 2u);
                        if _e535_ {
                            phi_1861_ = 0u;
                        } else {
                            let _e558_ = (_e84_ - 1u);
                            phi_1861_ = select(_e558_, _e557_, (_e557_ < _e558_));
                        }
                        let _e562_ = phi_1861_;
                        if (_e562_ < _e84_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e562_] = bitcast<i32>(_e531_.member);
                        let _e568_ = (_e468_ + 3u);
                        if _e535_ {
                            phi_1881_ = 0u;
                        } else {
                            let _e569_ = (_e84_ - 1u);
                            phi_1881_ = select(_e569_, _e568_, (_e568_ < _e569_));
                        }
                        let _e573_ = phi_1881_;
                        if (_e573_ < _e84_) {
                        } else {
                            break;
                        }
                        ozaki_buckets.member[_e573_] = bitcast<i32>(_e531_.member_1_);
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn ozaki_gram_rhs_i32(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
