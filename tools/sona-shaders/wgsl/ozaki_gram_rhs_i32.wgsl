struct type_6_ {
    member: array<f32>,
}

struct type_10_ {
    member: array<vec2<f32>>,
}

struct type_16_ {
    member: u32,
    member_1_: u32,
}

struct type_17_ {
    member: f32,
    member_1_: f32,
}

struct type_18_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: u32,
    member_6_: u32,
    member_7_: u32,
}

struct type_19_ {
    member: type_18_,
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
@group(0) @binding(0) 
var<uniform> uniforms: type_19_;
@group(0) @binding(1) 
var<storage> features: type_6_;
@group(0) @binding(2) 
var<storage> targets: type_6_;
@group(0) @binding(3) 
var<storage> col_scales: type_10_;
@group(0) @binding(4) 
var<storage, read_write> ozaki_buckets: type_22_;
var<workgroup> reduce_i32_: array<i32, 1280>;

fn function_() {
    var local: array<i32, 5>;
    var local_1_: array<i32, 5>;
    var phi_213_: u32;
    var phi_216_: u32;
    var phi_234_: u32;
    var phi_214_: u32;
    var phi_217_: u32;
    var local_2_: u32;
    var local_3_: u32;
    var local_4_: u32;
    var phi_240_: type_28_;
    var phi_241_: type_28_;
    var phi_242_: bool;
    var phi_247_: type_28_;
    var phi_166_: u32;
    var phi_169_: u32;
    var phi_191_: u32;
    var phi_167_: u32;
    var phi_170_: u32;
    var local_5_: u32;
    var local_6_: u32;
    var local_7_: u32;
    var local_8_: u32;
    var phi_248_: type_28_;
    var phi_249_: type_28_;
    var phi_250_: bool;
    var phi_254_: type_28_;
    var phi_255_: type_28_;
    var phi_267_: u32;
    var phi_289_: u32;
    var phi_331_: bool;
    var phi_332_: bool;
    var phi_333_: f32;
    var phi_335_: bool;
    var phi_336_: f32;
    var phi_337_: bool;
    var phi_338_: bool;
    var phi_345_: f32;
    var phi_1246_: bool;
    var phi_382_: u32;
    var phi_389_: type_17_;
    var phi_390_: type_17_;
    var phi_391_: bool;
    var phi_392_: type_17_;
    var phi_393_: bool;
    var phi_398_: type_17_;
    var phi_400_: f32;
    var phi_432_: bool;
    var phi_433_: bool;
    var phi_434_: f32;
    var phi_436_: bool;
    var phi_437_: f32;
    var phi_438_: bool;
    var phi_439_: bool;
    var phi_446_: f32;
    var phi_1261_: bool;
    var phi_483_: u32;
    var phi_490_: type_17_;
    var phi_491_: type_17_;
    var phi_492_: bool;
    var phi_493_: type_17_;
    var phi_494_: bool;
    var phi_499_: type_17_;
    var phi_501_: f32;
    var phi_508_: bool;
    var phi_511_: type_16_;
    var phi_514_: array<i32, 6>;
    var phi_512_: type_16_;
    var phi_537_: type_16_;
    var phi_572_: u32;
    var phi_618_: u32;
    var phi_597_: u32;
    var phi_626_: f32;
    var phi_701_: i32;
    var phi_731_: i32;
    var phi_761_: i32;
    var phi_764_: array<i32, 3>;
    var phi_765_: array<i32, 3>;
    var phi_767_: array<i32, 3>;
    var phi_768_: bool;
    var phi_773_: array<i32, 3>;
    var phi_848_: i32;
    var phi_878_: i32;
    var phi_908_: i32;
    var phi_911_: array<i32, 3>;
    var phi_912_: array<i32, 3>;
    var phi_914_: array<i32, 3>;
    var phi_915_: bool;
    var phi_920_: array<i32, 3>;
    var phi_967_: array<i32, 6>;
    var phi_968_: array<i32, 6>;
    var phi_969_: array<i32, 6>;
    var phi_515_: array<i32, 6>;
    var phi_1266_: bool;
    var local_9_: array<i32, 6>;
    var local_10_: array<i32, 6>;
    var local_11_: array<i32, 6>;
    var local_12_: array<i32, 6>;
    var local_13_: array<i32, 6>;
    var phi_984_: type_16_;
    var phi_985_: type_16_;
    var phi_1008_: type_16_;
    var phi_1363_: bool;
    var phi_1373_: bool;
    var phi_1043_: u32;
    var phi_1058_: type_16_;
    var phi_1059_: type_16_;
    var phi_1082_: type_16_;
    var phi_1369_: bool;
    var phi_1376_: bool;
    var phi_1375_: bool;
    var phi_1044_: u32;
    var phi_1374_: bool;
    var phi_1146_: type_16_;
    var phi_1147_: type_16_;
    var phi_1170_: type_16_;
    var phi_1196_: u32;
    var phi_1377_: bool;
    var phi_1214_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e66_ = wg_id_1_;
            let _e67_ = lid_1_;
            let _e69_ = arrayLength((&features.member));
            let _e71_ = arrayLength((&targets.member));
            let _e73_ = arrayLength((&col_scales.member));
            let _e75_ = arrayLength((&ozaki_buckets.member));
            let _e80_ = uniforms.member.member;
            let _e83_ = uniforms.member.member_1_;
            let _e86_ = uniforms.member.member_2_;
            let _e89_ = uniforms.member.member_3_;
            let _e91_ = select(_e89_, 1024u, (1024u < _e89_));
            let _e94_ = uniforms.member.member_4_;
            let _e96_ = select(1u, _e94_, (1u < _e94_));
            if (_e83_ == 0u) {
                phi_255_ = type_28_(0u, 0u, 0u, 0u, false, true);
            } else {
                let _e101_ = ((_e83_ * (_e83_ + 1u)) >> bitcast<u32>(1i));
                if (_e66_.x < _e101_) {
                    phi_166_ = (_e83_ - 1u);
                    phi_169_ = 0u;
                    loop {
                        let _e149_ = phi_166_;
                        let _e151_ = phi_169_;
                        local_5_ = _e151_;
                        local_6_ = _e151_;
                        local_7_ = _e151_;
                        local_8_ = _e151_;
                        let _e152_ = (_e151_ < _e149_);
                        if _e152_ {
                            let _e157_ = (_e151_ + (((_e149_ - _e151_) + 1u) >> bitcast<u32>(1i)));
                            let _e158_ = (_e83_ - _e157_);
                            let _e164_ = ((_e101_ - ((_e158_ * (_e158_ + 1u)) >> bitcast<u32>(1i))) <= _e66_.x);
                            if _e164_ {
                                phi_191_ = _e149_;
                            } else {
                                phi_191_ = (_e157_ - 1u);
                            }
                            let _e167_ = phi_191_;
                            phi_167_ = _e167_;
                            phi_170_ = select(_e151_, _e157_, _e164_);
                        } else {
                            phi_167_ = u32();
                            phi_170_ = u32();
                        }
                        let _e170_ = phi_167_;
                        let _e172_ = phi_170_;
                        phi_166_ = _e170_;
                        phi_169_ = _e172_;
                        if !(_e152_) {
                            break;
                        }
                    }
                    let _e175_ = local_5_;
                    let _e176_ = (_e83_ - _e175_);
                    let _e184_ = local_6_;
                    let _e185_ = (_e184_ + (_e66_.x - (_e101_ - ((_e176_ * (_e176_ + 1u)) >> bitcast<u32>(1i)))));
                    let _e187_ = local_7_;
                    let _e189_ = local_8_;
                    phi_248_ = type_28_(_e187_, _e185_, _e189_, _e185_, true, true);
                    phi_249_ = type_28_();
                    phi_250_ = true;
                } else {
                    let _e103_ = (_e66_.x - _e101_);
                    if (_e86_ == 0u) {
                        phi_241_ = type_28_();
                        phi_242_ = true;
                    } else {
                        let _e106_ = (_e103_ >= (_e83_ * _e86_));
                        if _e106_ {
                            phi_240_ = type_28_();
                        } else {
                            phi_213_ = (_e83_ - 1u);
                            phi_216_ = 0u;
                            loop {
                                let _e109_ = phi_213_;
                                let _e111_ = phi_216_;
                                local_2_ = _e111_;
                                local_3_ = _e111_;
                                local_4_ = _e111_;
                                let _e112_ = (_e111_ < _e109_);
                                if _e112_ {
                                    let _e117_ = (_e111_ + (((_e109_ - _e111_) + 1u) >> bitcast<u32>(1i)));
                                    let _e119_ = ((_e117_ * _e86_) <= _e103_);
                                    if _e119_ {
                                        phi_234_ = _e109_;
                                    } else {
                                        phi_234_ = (_e117_ - 1u);
                                    }
                                    let _e122_ = phi_234_;
                                    phi_214_ = _e122_;
                                    phi_217_ = select(_e111_, _e117_, _e119_);
                                } else {
                                    phi_214_ = u32();
                                    phi_217_ = u32();
                                }
                                let _e125_ = phi_214_;
                                let _e127_ = phi_217_;
                                phi_213_ = _e125_;
                                phi_216_ = _e127_;
                                if !(_e112_) {
                                    break;
                                }
                            }
                            let _e130_ = local_2_;
                            let _e132_ = (_e103_ - (_e130_ * _e86_));
                            let _e135_ = local_3_;
                            let _e137_ = local_4_;
                            phi_240_ = type_28_(_e135_, _e132_, _e137_, (_e83_ + _e132_), true, false);
                        }
                        let _e140_ = phi_240_;
                        phi_241_ = _e140_;
                        phi_242_ = _e106_;
                    }
                    let _e142_ = phi_241_;
                    let _e144_ = phi_242_;
                    if _e144_ {
                        phi_247_ = type_28_(0u, 0u, 0u, 0u, false, false);
                    } else {
                        phi_247_ = type_28_();
                    }
                    let _e146_ = phi_247_;
                    phi_248_ = _e146_;
                    phi_249_ = _e142_;
                    phi_250_ = _e144_;
                }
                let _e192_ = phi_248_;
                let _e194_ = phi_249_;
                let _e196_ = phi_250_;
                if _e196_ {
                    phi_254_ = _e192_;
                } else {
                    phi_254_ = _e194_;
                }
                let _e198_ = phi_254_;
                phi_255_ = _e198_;
            }
            let _e200_ = phi_255_;
            let _e202_ = (_e73_ == 0u);
            if _e202_ {
                phi_267_ = 0u;
            } else {
                let _e203_ = (_e73_ - 1u);
                phi_267_ = select(_e203_, _e200_.member_2_, (_e200_.member_2_ < _e203_));
            }
            let _e207_ = phi_267_;
            if (_e207_ < _e73_) {
            } else {
                break;
            }
            let _e212_ = col_scales.member[_e207_].x;
            let _e216_ = col_scales.member[_e207_].y;
            if _e202_ {
                phi_289_ = 0u;
            } else {
                let _e218_ = (_e73_ - 1u);
                phi_289_ = select(_e218_, _e200_.member_3_, (_e200_.member_3_ < _e218_));
            }
            let _e222_ = phi_289_;
            if (_e222_ < _e73_) {
            } else {
                break;
            }
            let _e227_ = col_scales.member[_e222_].x;
            let _e231_ = col_scales.member[_e222_].y;
            let _e232_ = bitcast<u32>(_e216_);
            let _e235_ = ((_e232_ >> bitcast<u32>(23u)) & 255u);
            if ((_e232_ & 2155872255u) == 0u) {
                let _e238_ = (_e235_ > 0u);
                if _e238_ {
                    let _e240_ = bitcast<u32>(_e212_);
                    let _e242_ = bitcast<f32>((_e240_ & 2147483647u));
                    if (_e235_ < 255u) {
                        if ((_e240_ & 2139095040u) != 2139095040u) {
                            phi_331_ = select(true, false, ((_e242_ * _e216_) <= 1f));
                        } else {
                            phi_331_ = true;
                        }
                        let _e249_ = phi_331_;
                        phi_332_ = _e249_;
                    } else {
                        phi_332_ = true;
                    }
                    let _e251_ = phi_332_;
                    phi_333_ = _e242_;
                    phi_335_ = _e251_;
                } else {
                    phi_333_ = f32();
                    phi_335_ = false;
                }
                let _e253_ = phi_333_;
                let _e255_ = phi_335_;
                phi_336_ = _e253_;
                phi_337_ = select(true, false, _e238_);
                phi_338_ = _e255_;
            } else {
                phi_336_ = f32();
                phi_337_ = true;
                phi_338_ = false;
            }
            let _e258_ = phi_336_;
            let _e260_ = phi_337_;
            let _e262_ = phi_338_;
            if _e260_ {
                phi_345_ = bitcast<f32>((bitcast<u32>(_e212_) & 2147483647u));
            } else {
                phi_345_ = _e258_;
            }
            let _e267_ = phi_345_;
            if select(_e262_, true, _e260_) {
                let _e269_ = bitcast<u32>(_e267_);
                if ((_e269_ & 2147483648u) == 0u) {
                    if (_e269_ == 0u) {
                        phi_390_ = type_17_();
                        phi_391_ = true;
                    } else {
                        let _e273_ = (_e269_ >= 2139095040u);
                        if _e273_ {
                            phi_389_ = type_17_();
                        } else {
                            if (_e267_ != _e267_) {
                                phi_1246_ = true;
                            } else {
                                phi_1246_ = (85070590000000000000000000000000000000f <= _e267_);
                            }
                            let _e277_ = phi_1246_;
                            let _e283_ = (((bitcast<u32>(select(_e267_, 85070590000000000000000000000000000000f, _e277_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                            if (_e283_ < 1u) {
                                phi_382_ = 1u;
                            } else {
                                phi_382_ = select(_e283_, 253u, (_e283_ > 253u));
                            }
                            let _e288_ = phi_382_;
                            phi_389_ = type_17_(bitcast<f32>((_e288_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e288_) << bitcast<u32>(23u))));
                        }
                        let _e298_ = phi_389_;
                        phi_390_ = _e298_;
                        phi_391_ = _e273_;
                    }
                    let _e300_ = phi_390_;
                    let _e302_ = phi_391_;
                    phi_392_ = _e300_;
                    phi_393_ = _e302_;
                } else {
                    phi_392_ = type_17_();
                    phi_393_ = true;
                }
                let _e304_ = phi_392_;
                let _e306_ = phi_393_;
                if _e306_ {
                    phi_398_ = type_17_(1f, 1f);
                } else {
                    phi_398_ = _e304_;
                }
                let _e308_ = phi_398_;
                phi_400_ = _e308_.member_1_;
            } else {
                phi_400_ = _e216_;
            }
            let _e311_ = phi_400_;
            let _e312_ = bitcast<u32>(_e231_);
            let _e315_ = ((_e312_ >> bitcast<u32>(23u)) & 255u);
            if ((_e312_ & 2155872255u) == 0u) {
                let _e318_ = (_e315_ > 0u);
                if _e318_ {
                    let _e320_ = bitcast<u32>(_e227_);
                    let _e322_ = bitcast<f32>((_e320_ & 2147483647u));
                    if (_e315_ < 255u) {
                        if ((_e320_ & 2139095040u) != 2139095040u) {
                            phi_432_ = select(true, false, ((_e322_ * _e231_) <= 1f));
                        } else {
                            phi_432_ = true;
                        }
                        let _e329_ = phi_432_;
                        phi_433_ = _e329_;
                    } else {
                        phi_433_ = true;
                    }
                    let _e331_ = phi_433_;
                    phi_434_ = _e322_;
                    phi_436_ = _e331_;
                } else {
                    phi_434_ = f32();
                    phi_436_ = false;
                }
                let _e333_ = phi_434_;
                let _e335_ = phi_436_;
                phi_437_ = _e333_;
                phi_438_ = select(true, false, _e318_);
                phi_439_ = _e335_;
            } else {
                phi_437_ = f32();
                phi_438_ = true;
                phi_439_ = false;
            }
            let _e338_ = phi_437_;
            let _e340_ = phi_438_;
            let _e342_ = phi_439_;
            if _e340_ {
                phi_446_ = bitcast<f32>((bitcast<u32>(_e227_) & 2147483647u));
            } else {
                phi_446_ = _e338_;
            }
            let _e347_ = phi_446_;
            if select(_e342_, true, _e340_) {
                let _e349_ = bitcast<u32>(_e347_);
                if ((_e349_ & 2147483648u) == 0u) {
                    if (_e349_ == 0u) {
                        phi_491_ = type_17_();
                        phi_492_ = true;
                    } else {
                        let _e353_ = (_e349_ >= 2139095040u);
                        if _e353_ {
                            phi_490_ = type_17_();
                        } else {
                            if (_e347_ != _e347_) {
                                phi_1261_ = true;
                            } else {
                                phi_1261_ = (85070590000000000000000000000000000000f <= _e347_);
                            }
                            let _e357_ = phi_1261_;
                            let _e363_ = (((bitcast<u32>(select(_e347_, 85070590000000000000000000000000000000f, _e357_)) + 8388607u) & 2139095040u) >> bitcast<u32>(23u));
                            if (_e363_ < 1u) {
                                phi_483_ = 1u;
                            } else {
                                phi_483_ = select(_e363_, 253u, (_e363_ > 253u));
                            }
                            let _e368_ = phi_483_;
                            phi_490_ = type_17_(bitcast<f32>((_e368_ << bitcast<u32>(23u))), bitcast<f32>(((254u - _e368_) << bitcast<u32>(23u))));
                        }
                        let _e378_ = phi_490_;
                        phi_491_ = _e378_;
                        phi_492_ = _e353_;
                    }
                    let _e380_ = phi_491_;
                    let _e382_ = phi_492_;
                    phi_493_ = _e380_;
                    phi_494_ = _e382_;
                } else {
                    phi_493_ = type_17_();
                    phi_494_ = true;
                }
                let _e384_ = phi_493_;
                let _e386_ = phi_494_;
                if _e386_ {
                    phi_499_ = type_17_(1f, 1f);
                } else {
                    phi_499_ = _e384_;
                }
                let _e388_ = phi_499_;
                phi_501_ = _e388_.member_1_;
            } else {
                phi_501_ = _e231_;
            }
            let _e391_ = phi_501_;
            if _e200_.member_4_ {
                phi_508_ = (_e66_.y < _e96_);
            } else {
                phi_508_ = false;
            }
            let _e396_ = phi_508_;
            phi_511_ = type_16_(0u, 4u);
            phi_514_ = array<i32, 6>(0i, 0i, 0i, 0i, 0i, 0i);
            loop {
                let _e398_ = phi_511_;
                let _e400_ = phi_514_;
                local_9_ = _e400_;
                local_10_ = _e400_;
                local_11_ = _e400_;
                local_12_ = _e400_;
                local_13_ = _e400_;
                if (_e398_.member < _e398_.member_1_) {
                    phi_512_ = type_16_((_e398_.member + 1u), _e398_.member_1_);
                    phi_537_ = type_16_(1u, _e398_.member);
                } else {
                    phi_512_ = _e398_;
                    phi_537_ = type_16_(0u, u32());
                }
                let _e413_ = phi_512_;
                let _e415_ = phi_537_;
                let _e419_ = (bitcast<i32>(_e415_.member) != 0i);
                if _e419_ {
                    let _e421_ = ((_e415_.member_1_ * 256u) + _e67_);
                    let _e422_ = ((_e66_.y * _e91_) + _e421_);
                    if _e396_ {
                        if (_e421_ < _e91_) {
                            if (_e422_ < _e80_) {
                                let _e427_ = ((_e200_.member * _e80_) + _e422_);
                                let _e428_ = (_e69_ == 0u);
                                if _e428_ {
                                    phi_572_ = 0u;
                                } else {
                                    let _e429_ = (_e69_ - 1u);
                                    phi_572_ = select(_e429_, _e427_, (_e427_ < _e429_));
                                }
                                let _e433_ = phi_572_;
                                if (_e433_ < _e69_) {
                                } else {
                                    phi_1266_ = true;
                                    break;
                                }
                                let _e437_ = features.member[_e433_];
                                if _e200_.member_5_ {
                                    let _e454_ = ((_e200_.member_1_ * _e80_) + _e422_);
                                    if _e428_ {
                                        phi_597_ = 0u;
                                    } else {
                                        let _e455_ = (_e69_ - 1u);
                                        phi_597_ = select(_e455_, _e454_, (_e454_ < _e455_));
                                    }
                                    let _e459_ = phi_597_;
                                    if (_e459_ < _e69_) {
                                    } else {
                                        phi_1266_ = true;
                                        break;
                                    }
                                    let _e463_ = features.member[_e459_];
                                    phi_626_ = _e463_;
                                } else {
                                    let _e441_ = ((_e200_.member_1_ * _e80_) + _e422_);
                                    if (_e71_ == 0u) {
                                        phi_618_ = 0u;
                                    } else {
                                        let _e443_ = (_e71_ - 1u);
                                        phi_618_ = select(_e443_, _e441_, (_e441_ < _e443_));
                                    }
                                    let _e447_ = phi_618_;
                                    if (_e447_ < _e71_) {
                                    } else {
                                        phi_1266_ = true;
                                        break;
                                    }
                                    let _e451_ = targets.member[_e447_];
                                    phi_626_ = _e451_;
                                }
                                let _e465_ = phi_626_;
                                if ((bitcast<u32>(_e437_) & 2139095040u) != 2139095040u) {
                                    let _e471_ = ((bitcast<u32>(_e311_) & 2139095040u) != 2139095040u);
                                    if _e471_ {
                                        let _e473_ = select(_e437_, -85070590000000000000000000000000000000f, (_e437_ < -85070590000000000000000000000000000000f));
                                        let _e476_ = (select(_e473_, 85070590000000000000000000000000000000f, (_e473_ > 85070590000000000000000000000000000000f)) * _e311_);
                                        if ((bitcast<u32>(_e476_) & 2139095040u) != 2139095040u) {
                                            let _e481_ = select(_e476_, -1f, (_e476_ < -1f));
                                            let _e483_ = select(_e481_, 1f, (_e481_ > 1f));
                                            let _e484_ = (_e483_ * 256f);
                                            if ((bitcast<u32>(_e484_) & 2139095040u) != 2139095040u) {
                                                let _e489_ = select(_e484_, -4194304f, (_e484_ < -4194304f));
                                                phi_701_ = (bitcast<i32>(bitcast<u32>((select(_e489_, 4194304f, (_e489_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_701_ = 0i;
                                            }
                                            let _e497_ = phi_701_;
                                            let _e500_ = (_e483_ - (f32(_e497_) * 0.00390625f));
                                            let _e501_ = (_e500_ * 65536f);
                                            if ((bitcast<u32>(_e501_) & 2139095040u) != 2139095040u) {
                                                let _e506_ = select(_e501_, -4194304f, (_e501_ < -4194304f));
                                                phi_731_ = (bitcast<i32>(bitcast<u32>((select(_e506_, 4194304f, (_e506_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_731_ = 0i;
                                            }
                                            let _e514_ = phi_731_;
                                            let _e518_ = ((_e500_ - (f32(_e514_) * 0.000015258789f)) * 16777216f);
                                            if ((bitcast<u32>(_e518_) & 2139095040u) != 2139095040u) {
                                                let _e523_ = select(_e518_, -4194304f, (_e518_ < -4194304f));
                                                phi_761_ = (bitcast<i32>(bitcast<u32>((select(_e523_, 4194304f, (_e523_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_761_ = 0i;
                                            }
                                            let _e531_ = phi_761_;
                                            phi_764_ = array<i32, 3>(_e497_, _e514_, _e531_);
                                        } else {
                                            phi_764_ = array<i32, 3>(0i, 0i, 0i);
                                        }
                                        let _e534_ = phi_764_;
                                        phi_765_ = _e534_;
                                    } else {
                                        phi_765_ = array<i32, 3>();
                                    }
                                    let _e536_ = phi_765_;
                                    phi_767_ = _e536_;
                                    phi_768_ = select(true, false, _e471_);
                                } else {
                                    phi_767_ = array<i32, 3>();
                                    phi_768_ = true;
                                }
                                let _e539_ = phi_767_;
                                let _e541_ = phi_768_;
                                if _e541_ {
                                    phi_773_ = array<i32, 3>(0i, 0i, 0i);
                                } else {
                                    phi_773_ = _e539_;
                                }
                                let _e543_ = phi_773_;
                                if ((bitcast<u32>(_e465_) & 2139095040u) != 2139095040u) {
                                    let _e549_ = ((bitcast<u32>(_e391_) & 2139095040u) != 2139095040u);
                                    if _e549_ {
                                        let _e551_ = select(_e465_, -85070590000000000000000000000000000000f, (_e465_ < -85070590000000000000000000000000000000f));
                                        let _e554_ = (select(_e551_, 85070590000000000000000000000000000000f, (_e551_ > 85070590000000000000000000000000000000f)) * _e391_);
                                        if ((bitcast<u32>(_e554_) & 2139095040u) != 2139095040u) {
                                            let _e559_ = select(_e554_, -1f, (_e554_ < -1f));
                                            let _e561_ = select(_e559_, 1f, (_e559_ > 1f));
                                            let _e562_ = (_e561_ * 256f);
                                            if ((bitcast<u32>(_e562_) & 2139095040u) != 2139095040u) {
                                                let _e567_ = select(_e562_, -4194304f, (_e562_ < -4194304f));
                                                phi_848_ = (bitcast<i32>(bitcast<u32>((select(_e567_, 4194304f, (_e567_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_848_ = 0i;
                                            }
                                            let _e575_ = phi_848_;
                                            let _e578_ = (_e561_ - (f32(_e575_) * 0.00390625f));
                                            let _e579_ = (_e578_ * 65536f);
                                            if ((bitcast<u32>(_e579_) & 2139095040u) != 2139095040u) {
                                                let _e584_ = select(_e579_, -4194304f, (_e579_ < -4194304f));
                                                phi_878_ = (bitcast<i32>(bitcast<u32>((select(_e584_, 4194304f, (_e584_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_878_ = 0i;
                                            }
                                            let _e592_ = phi_878_;
                                            let _e596_ = ((_e578_ - (f32(_e592_) * 0.000015258789f)) * 16777216f);
                                            if ((bitcast<u32>(_e596_) & 2139095040u) != 2139095040u) {
                                                let _e601_ = select(_e596_, -4194304f, (_e596_ < -4194304f));
                                                phi_908_ = (bitcast<i32>(bitcast<u32>((select(_e601_, 4194304f, (_e601_ > 4194304f)) + 12582912f))) - 1262485504i);
                                            } else {
                                                phi_908_ = 0i;
                                            }
                                            let _e609_ = phi_908_;
                                            phi_911_ = array<i32, 3>(_e575_, _e592_, _e609_);
                                        } else {
                                            phi_911_ = array<i32, 3>(0i, 0i, 0i);
                                        }
                                        let _e612_ = phi_911_;
                                        phi_912_ = _e612_;
                                    } else {
                                        phi_912_ = array<i32, 3>();
                                    }
                                    let _e614_ = phi_912_;
                                    phi_914_ = _e614_;
                                    phi_915_ = select(true, false, _e549_);
                                } else {
                                    phi_914_ = array<i32, 3>();
                                    phi_915_ = true;
                                }
                                let _e617_ = phi_914_;
                                let _e619_ = phi_915_;
                                if _e619_ {
                                    phi_920_ = array<i32, 3>(0i, 0i, 0i);
                                } else {
                                    phi_920_ = _e617_;
                                }
                                let _e621_ = phi_920_;
                                let _e633_ = array<i32, 6>((_e400_[0] + (_e543_[0] * _e621_[0])), _e400_[1], _e400_[2], _e400_[3], _e400_[4], _e400_[5]);
                                let _e647_ = array<i32, 6>(_e633_[0], (_e400_[1] + ((_e543_[0] * _e621_[1]) + (_e543_[1] * _e621_[0]))), _e633_[2], _e633_[3], _e633_[4], _e633_[5]);
                                let _e663_ = array<i32, 6>(_e647_[0], _e647_[1], (_e400_[2] + (((_e543_[0] * _e621_[2]) + (_e543_[1] * _e621_[1])) + (_e543_[2] * _e621_[0]))), _e647_[3], _e647_[4], _e647_[5]);
                                let _e675_ = array<i32, 6>(_e663_[0], _e663_[1], _e663_[2], (_e400_[3] + ((_e543_[1] * _e621_[2]) + (_e543_[2] * _e621_[1]))), _e663_[4], _e663_[5]);
                                phi_967_ = array<i32, 6>(_e675_[0], _e675_[1], _e675_[2], _e675_[3], (_e400_[4] + (_e543_[2] * _e621_[2])), _e675_[5]);
                            } else {
                                phi_967_ = _e400_;
                            }
                            let _e687_ = phi_967_;
                            phi_968_ = _e687_;
                        } else {
                            phi_968_ = _e400_;
                        }
                        let _e689_ = phi_968_;
                        phi_969_ = _e689_;
                    } else {
                        phi_969_ = _e400_;
                    }
                    let _e691_ = phi_969_;
                    phi_515_ = _e691_;
                } else {
                    phi_515_ = array<i32, 6>();
                }
                let _e693_ = phi_515_;
                phi_511_ = _e413_;
                phi_514_ = _e693_;
                phi_1266_ = false;
                if !(_e419_) {
                    break;
                }
            }
            let _e696_ = phi_1266_;
            if _e696_ {
                break;
            }
            let _e698_ = local_9_;
            let _e701_ = local_10_;
            let _e704_ = local_11_;
            let _e707_ = local_12_;
            let _e710_ = local_13_;
            local = array<i32, 5>(_e698_[0], _e701_[1], _e704_[2], _e707_[3], _e710_[4]);
            phi_984_ = type_16_(0u, 5u);
            loop {
                let _e714_ = phi_984_;
                if (_e714_.member < _e714_.member_1_) {
                    phi_985_ = type_16_((_e714_.member + 1u), _e714_.member_1_);
                    phi_1008_ = type_16_(1u, _e714_.member);
                } else {
                    phi_985_ = _e714_;
                    phi_1008_ = type_16_(0u, u32());
                }
                let _e727_ = phi_985_;
                let _e729_ = phi_1008_;
                let _e733_ = (bitcast<i32>(_e729_.member) != 0i);
                if _e733_ {
                    if (_e729_.member_1_ < 5u) {
                    } else {
                        phi_1363_ = true;
                        break;
                    }
                    let _e736_ = local[_e729_.member_1_];
                    let _e738_ = ((_e729_.member_1_ * 256u) + _e67_);
                    let _e740_ = select(1279u, _e738_, (_e738_ < 1279u));
                    if (_e740_ < 1280u) {
                    } else {
                        phi_1363_ = true;
                        break;
                    }
                    reduce_i32_[_e740_] = _e736_;
                }
                phi_984_ = _e727_;
                phi_1363_ = _e696_;
                if !(_e733_) {
                    break;
                }
            }
            let _e745_ = phi_1363_;
            if _e745_ {
                break;
            }
            workgroupBarrier();
            phi_1373_ = _e745_;
            phi_1043_ = 128u;
            loop {
                let _e747_ = phi_1373_;
                let _e749_ = phi_1043_;
                let _e750_ = (_e749_ > 0u);
                if _e750_ {
                    if (_e67_ < _e749_) {
                        phi_1058_ = type_16_(0u, 5u);
                        loop {
                            let _e753_ = phi_1058_;
                            if (_e753_.member < _e753_.member_1_) {
                                phi_1059_ = type_16_((_e753_.member + 1u), _e753_.member_1_);
                                phi_1082_ = type_16_(1u, _e753_.member);
                            } else {
                                phi_1059_ = _e753_;
                                phi_1082_ = type_16_(0u, u32());
                            }
                            let _e766_ = phi_1059_;
                            let _e768_ = phi_1082_;
                            let _e772_ = (bitcast<i32>(_e768_.member) != 0i);
                            if _e772_ {
                                let _e774_ = ((_e768_.member_1_ * 256u) + _e67_);
                                let _e775_ = (_e774_ < 1280u);
                                if _e775_ {
                                } else {
                                    phi_1369_ = true;
                                    break;
                                }
                                let _e777_ = reduce_i32_[_e774_];
                                let _e778_ = (_e774_ + _e749_);
                                if (_e778_ < 1280u) {
                                } else {
                                    phi_1369_ = true;
                                    break;
                                }
                                let _e781_ = reduce_i32_[_e778_];
                                if _e775_ {
                                } else {
                                    phi_1369_ = true;
                                    break;
                                }
                                reduce_i32_[_e774_] = (_e777_ + _e781_);
                            }
                            phi_1058_ = _e766_;
                            phi_1369_ = _e747_;
                            if !(_e772_) {
                                break;
                            }
                        }
                        let _e785_ = phi_1369_;
                        phi_1374_ = _e785_;
                        if _e785_ {
                            break;
                        }
                        phi_1376_ = _e785_;
                    } else {
                        phi_1376_ = _e747_;
                    }
                    let _e787_ = phi_1376_;
                    workgroupBarrier();
                    phi_1375_ = _e787_;
                    phi_1044_ = (_e749_ >> bitcast<u32>(1i));
                } else {
                    phi_1375_ = _e747_;
                    phi_1044_ = u32();
                }
                let _e791_ = phi_1375_;
                let _e793_ = phi_1044_;
                phi_1373_ = _e791_;
                phi_1043_ = _e793_;
                phi_1374_ = _e791_;
                if !(_e750_) {
                    break;
                }
            }
            let _e796_ = phi_1374_;
            if _e796_ {
                break;
            }
            let _e798_ = reduce_i32_[0];
            let _e800_ = reduce_i32_[256];
            let _e802_ = reduce_i32_[512];
            let _e804_ = reduce_i32_[768];
            let _e806_ = reduce_i32_[1024];
            workgroupBarrier();
            local_1_ = array<i32, 5>(_e798_, _e800_, _e802_, _e804_, _e806_);
            if (_e67_ == 0u) {
                if _e396_ {
                    let _e811_ = (((_e66_.x * _e96_) + _e66_.y) * 6u);
                    phi_1146_ = type_16_(0u, 5u);
                    loop {
                        let _e813_ = phi_1146_;
                        if (_e813_.member < _e813_.member_1_) {
                            phi_1147_ = type_16_((_e813_.member + 1u), _e813_.member_1_);
                            phi_1170_ = type_16_(1u, _e813_.member);
                        } else {
                            phi_1147_ = _e813_;
                            phi_1170_ = type_16_(0u, u32());
                        }
                        let _e826_ = phi_1147_;
                        let _e828_ = phi_1170_;
                        let _e832_ = (bitcast<i32>(_e828_.member) != 0i);
                        if _e832_ {
                            if (_e828_.member_1_ < 5u) {
                            } else {
                                phi_1377_ = true;
                                break;
                            }
                            let _e835_ = local_1_[_e828_.member_1_];
                            let _e836_ = (_e811_ + _e828_.member_1_);
                            if (_e75_ == 0u) {
                                phi_1196_ = 0u;
                            } else {
                                let _e838_ = (_e75_ - 1u);
                                phi_1196_ = select(_e838_, _e836_, (_e836_ < _e838_));
                            }
                            let _e842_ = phi_1196_;
                            if (_e842_ < _e75_) {
                            } else {
                                phi_1377_ = true;
                                break;
                            }
                            ozaki_buckets.member[_e842_] = _e835_;
                        }
                        phi_1146_ = _e826_;
                        phi_1377_ = _e796_;
                        if !(_e832_) {
                            break;
                        }
                    }
                    let _e848_ = phi_1377_;
                    if _e848_ {
                        break;
                    }
                    let _e849_ = (_e811_ + 5u);
                    if (_e75_ == 0u) {
                        phi_1214_ = 0u;
                    } else {
                        let _e851_ = (_e75_ - 1u);
                        phi_1214_ = select(_e851_, _e849_, (_e849_ < _e851_));
                    }
                    let _e855_ = phi_1214_;
                    if (_e855_ < _e75_) {
                    } else {
                        break;
                    }
                    ozaki_buckets.member[_e855_] = 0i;
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn ozaki_gram_rhs_i32(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}
